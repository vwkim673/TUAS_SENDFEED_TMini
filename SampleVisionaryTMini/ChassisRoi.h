#pragma once
// ============================================================================
// ChassisRoi.h
//
// ROI-based chassis backup for when VA (2D model) is unavailable/blind.
// Gated by FORCE_ROI_MODE (process.ini).
//
// Design (revised 2026-06-13, after run #1 found the WHEEL, not the cone):
// Do NOT reinvent target finding. The chassis target is defined exactly as the
// VA path defines it: crop a region, then `pc_cone_detection_naive` (etc.)
// isolates the cone by NEAREST-Z depth slab [minZ-100, minZ+300] + trolley
// corner. The only thing the backup changes is the CROP SOURCE: instead of the
// VA detection box, use the per-job preset world ROI (presetRangeX x height x
// gantry from INI).
//
//   chassis_roi_crop()        -> preset-ROI crop (drop invalids). Height is the
//                                cone band (excludes wheel/ground); trolley is
//                                the per-job preset (±~200mm, lane-bordered).
//   chassis_roi_detect_cone() -> cone = NEAREST-Z protrusion in the crop. The
//                                twist-lock cone sticks toward the side-mounted
//                                sensor, so a THIN near-Z slab (~100mm) isolates
//                                it from the continuous deck behind it. Validated
//                                2026-06-14 (~20mm vs VA). NOTE: delegating to
//                                pc_cone_detection_naive (deep 300mm slab +
//                                trolley corner) FAILED here — the deck dominates
//                                and the corner grabs the wrong trolley edge.
//
//   ChassisRoiTracker   -> watches the DETECTED cone position (the output of
//                          pc_cone_detection_naive) for lock + driver-creep,
//                          NOT a separate localization. Chassis is stationary
//                          until load transfer, so: ACQUIRING -> LOCK after K
//                          stable detections; LOCKED holds the baseline and
//                          flags CHASSIS_MOVED on sustained lateral drift;
//                          frozen while the spreader is landed.
//
// Debug PCDs (savePath != ""):  _<tag>_roiCrop.pcd  (crop handed to detector).
// ============================================================================

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

struct ChassisRoiParams
{
	//Broad search ROI in world mm (per-job preset from INI).
	float xMin = -10000.f, xMax = 10000.f;	//trolley (T)
	float yMin = -10000.f, yMax = 10000.f;	//height  (H), up = +Y
	float zMin = -10000.f, zMax = 10000.f;	//gantry  (G)

	//Cone detection: the twist-lock cone protrudes TOWARD the sensor (nearest
	//Z). A THIN near-Z slab isolates it from the bulkier deck behind it (a deep
	//300mm slab lets the deck dominate). 2026-06-14: validated ~20mm vs VA.
	float zSlab = 100.f;		//near-Z slab depth: [minZ-zPad, minZ+zSlab]
	float zPad = 30.f;		//include a little before the dense minZ
	int minConePts = 30;		//min points to accept a cone

	//Tracking (operates on the DETECTED cone position).
	int lockK = 3;			//consecutive agreeing detections to lock
	float lockTol = 30.f;		//acquire agreement tolerance (mm)
	float moveTol = 50.f;		//lateral move vs baseline => chassis moved (mm)
	int moveConfirm = 3;		//consecutive drift frames to confirm a move
	int maxMiss = 15;		//consecutive missed detections => unlock

	//--- Stage-2 REGION extraction (port of tools/roi_cone_extract.py) ---------
	//Axes here: p.x = lateral (Gantry, side-rule axis) ; p.y = height (up=+) ;
	//p.z = depth (Trolley, the cone protrudes = nearest-Z). This isolates the
	//target cone+plate REGION so it can replace the VA-mapped pcCone before PCA.
	float coneY       = -10000.f;	//PER-CONFIG cone height (world Y), from process.ini preset.
					//Required: the CHS20 wheel is nearer-Z than the cone, so a
					//runtime nearest-Z estimate would lock onto the wheel. -10000
					//= unset -> falls back to a (less reliable) runtime estimate.
	bool  guideDenseY = false;	//IGH: centre candBand on the DOMINANT dense-Y mass (the guide)
					//instead of coneY. The twist-lock cone/post sits ~250mm ABOVE the
					//guide and is nearer-Z, so coneY / near-Z centring targets the WRONG
					//(upper) structure; the guide is the largest mass in the crop.
	float guideDenseWin = 120.f;	//height window (mm) for the dominant-mass search
	float guideDeckGuard = 150.f;	//exclude points within this of the crop floor (min-Y): a deck/
					//funnel-bottom mass can out-density the guide and pull the band down.
	float candYband   = 140.f;	//candidate cone-height band (drops the low handle/rail)
	float colXbin     = 40.f;	//lateral bin width for per-column near-Z
	float protrudeTol = 100.f;	//a column is "cone" if nearZ <= globalMin+this
	int   minColPts   = 60;		//min points for a column to count (rejects sparse noise)
	float coneGap     = 70.f;	//lateral gap that separates two cones
	float colSlab     = 130.f;	//candidate near-Z slab depth (for the x-position only)
	float xspan       = 300.f;	//tight lateral window around the cone nose
	float zDepth      = 200.f;	//cone+plate depth inward from the front face
	float zBinw       = 25.f;	//depth histogram bin
	float zAbsFloor   = 10.f;	//depth noise floor (below = not cone)
	float zMaxGap     = 55.f;	//depth front walk-back gap-hop
	float zYband      = 200.f;	//height band used when measuring the depth band
	float yBinw       = 30.f;	//height histogram bin
	float yReachDown  = 230.f;	//height cluster reach below cy (plate + below)
	float yReachUp    = 130.f;	//height cluster reach above cy (cone top)
	float guideReachUp = 130.f;	//IGH ONLY (guideDenseY): reach UP above the dense guide centre.
					//SENSOR-SPECIFIC (set in fillRoiParams): FRONT=240 -- the guide top-left
					//corner sits ~200mm above cy AND the twist-lock POST is OCCLUDED, so the
					//XT-sized 130 CLIPS the corner (~80mm H under-read) and extending is safe.
					//REAR=130 (default) -- the corner is only ~100mm up but the POST is VISIBLE
					//and continuous, so a larger reach climbs the post to the cone (+180mm).
	float yBelow      = 100.f;	//keep this far under the plate (capped at wheel)
	float yMaxGap     = 45.f;	//height cluster gap-hop
	float yPeakFrac   = 0.2f;	//height occupancy threshold = frac * cone-region peak
	int   yMinAbs     = 10;		//height occupancy absolute floor
};

namespace chassis_roi_detail
{
	//Dense extreme of a value list (ignores isolated flyers): mean of the first
	//window (>= minPts within windowSize) from the low (or high) end.
	inline float denseExtreme(std::vector<float> vals, float windowSize, int minPts, bool findMax)
	{
		if (vals.empty()) return findMax ? -1e9f : 1e9f;
		std::sort(vals.begin(), vals.end());
		if ((int)vals.size() < minPts) return findMax ? vals.back() : vals.front();
		if (findMax)
		{
			int hi = (int)vals.size() - 1, lo = hi;
			while (hi >= 0)
			{
				while (lo > 0 && vals[hi] - vals[lo - 1] <= windowSize) lo--;
				if (hi - lo + 1 >= minPts)
				{
					float s = 0.f; for (int i = lo; i <= hi; i++) s += vals[i];
					return s / (float)(hi - lo + 1);
				}
				hi--; if (lo > hi) lo = hi;
			}
			return vals.back();
		}
		int lo = 0, hi = 0;
		while (lo < (int)vals.size())
		{
			while (hi < (int)vals.size() - 1 && vals[hi + 1] - vals[lo] <= windowSize) hi++;
			if (hi - lo + 1 >= minPts)
			{
				float s = 0.f; for (int i = lo; i <= hi; i++) s += vals[i];
				return s / (float)(hi - lo + 1);
			}
			lo++; if (hi < lo) hi = lo;
		}
		return vals.front();
	}
}

//Detect the chassis cone/guide as the NEAREST-Z protrusion inside the crop.
//The crop is already limited to the cone-height band + trolley ROI; the cone
//sticks up toward the side-mounted sensor, so it is the nearest-Z structure.
//Returns the cone centroid, or (-10000,...) if not found.
inline pcl::PointXYZ chassis_roi_detect_cone(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& crop,
	const ChassisRoiParams& prm,
	const std::string& savePath = std::string(""),
	const std::string& saveTag = std::string("ROI"))
{
	pcl::PointXYZ none(-10000.f, -10000.f, -10000.f);
	if (!crop || (int)crop->points.size() < prm.minConePts) return none;

	//Robust nearest Z (dense min ignores single flyer points).
	std::vector<float> zs; zs.reserve(crop->points.size());
	for (const auto& p : crop->points) zs.push_back(p.z);
	float zNear = chassis_roi_detail::denseExtreme(zs, 40.f, prm.minConePts, false);

	//Thin near-Z slab = the protruding cone.
	pcl::PointCloud<pcl::PointXYZ>::Ptr slab(new pcl::PointCloud<pcl::PointXYZ>);
	for (const auto& p : crop->points)
		if (p.z >= zNear - prm.zPad && p.z <= zNear + prm.zSlab) slab->points.push_back(p);

	if ((int)slab->points.size() < prm.minConePts) return none;

	double sx = 0, sy = 0, sz = 0;
	for (const auto& p : slab->points) { sx += p.x; sy += p.y; sz += p.z; }
	int n = (int)slab->points.size();
	pcl::PointXYZ cone((float)(sx / n), (float)(sy / n), (float)(sz / n));

	if (!savePath.empty())
	{
		try
		{
			slab->width = (uint32_t)n; slab->height = 1; slab->is_dense = true;
			pcl::io::savePCDFileBinaryCompressed(savePath + "_" + saveTag + "_coneSlab.pcd", *slab);
			pcl::PointCloud<pcl::PointXYZ> cc; cc.points.push_back(cone);
			cc.width = 1; cc.height = 1; cc.is_dense = true;
			pcl::io::savePCDFileBinaryCompressed(savePath + "_" + saveTag + "_conePos.pcd", cc);
		}
		catch (...) {}
	}
	return cone;
}

// ============================================================================
// Stage-2 REGION extraction (port of tools/roi_cone_extract.py).
// Isolates the target cone+plate REGION inside the preset crop so it can replace
// the VA-mapped pcCone (then position_detection_pca runs on it as usual).
// Stages: estimate cone height -> per-lateral-column near-Z candidates -> side-
// rule select -> tight box (lateral xspan / depth front+zDepth / height cluster).
// ============================================================================
namespace chassis_roi_detail
{
	inline float medianOf(std::vector<float> v)
	{
		if (v.empty()) return 0.f;
		std::sort(v.begin(), v.end());
		size_t n = v.size();
		return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
	}

	//Cone height = Y centroid of the nearest-Z slab (the protruding cone), used to
	//centre the candidate band and the height cluster.
	inline float coneHeightEstimate(const pcl::PointCloud<pcl::PointXYZ>::Ptr& crop, const ChassisRoiParams& prm)
	{
		std::vector<float> zs; zs.reserve(crop->points.size());
		for (const auto& p : crop->points) zs.push_back(p.z);
		float zNear = denseExtreme(zs, 40.f, prm.minConePts, false);
		double sy = 0; int n = 0;
		for (const auto& p : crop->points)
			if (p.z >= zNear - prm.zPad && p.z <= zNear + prm.zSlab) { sy += p.y; n++; }
		return n > 0 ? (float)(sy / n) : 0.f;
	}

	//Y centre of the DENSEST `win`-mm height window = the dominant structure. For IGH the
	//GUIDE is the largest mass; the twist-lock cone/post above it (nearer-Z, smaller) would
	//be picked by coneY / near-Z centring. Sliding-window max-count over sorted Y.
	inline float dominantDenseY(const pcl::PointCloud<pcl::PointXYZ>::Ptr& c, float win, float deckGuard = 0.f)
	{
		std::vector<float> ys; ys.reserve(c->points.size());
		for (const auto& p : c->points) ys.push_back(p.y);
		if (ys.empty()) return 0.f;
		std::sort(ys.begin(), ys.end());
		//Drop the deck floor: a flat deck / funnel-bottom near the crop's min-Y can be DENSER
		//than the guide and would win the dominant-mass search, pulling the band ~300mm too low
		//(seen on FRONT CHS45 Big). Exclude points within deckGuard of min-Y (keep if enough
		//remain) so the guide wins.
		if (deckGuard > 0.f)
		{
			float floorY = ys.front() + deckGuard;
			size_t k = 0; while (k < ys.size() && ys[k] < floorY) k++;
			if ((int)(ys.size() - k) >= 30) ys.erase(ys.begin(), ys.begin() + (long)k);
		}
		int n = (int)ys.size(), lo = 0, bestN = 0; float bestC = ys[0];
		for (int hi = 0; hi < n; hi++)
		{
			while (ys[hi] - ys[lo] > win) lo++;
			if (hi - lo + 1 > bestN) { bestN = hi - lo + 1; bestC = 0.5f * (ys[lo] + ys[hi]); }
		}
		return bestC;
	}

	//Per-lateral-column near-Z protrusion -> one candidate cloud per cone. Each
	//column keeps its OWN near-Z so a deeper target isn't lost behind a nearer
	//partner; deck between cones (farther Z) drops out, splitting them.
	inline std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> coneColumns(
		const pcl::PointCloud<pcl::PointXYZ>::Ptr& crop, const ChassisRoiParams& prm)
	{
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> out;
		if (!crop || (int)crop->points.size() < prm.minConePts) return out;
		float xmin = 1e9f, xmax = -1e9f;
		for (const auto& p : crop->points) { xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x); }
		int nb = std::max(1, (int)std::ceil((xmax - xmin) / prm.colXbin)) + 1;
		std::vector<std::vector<float>> binZ(nb);
		for (const auto& p : crop->points)
		{
			int bi = (int)((p.x - xmin) / prm.colXbin);
			if (bi < 0) bi = 0; if (bi >= nb) bi = nb - 1;
			binZ[bi].push_back(p.z);
		}
		std::vector<float> qCen, qNZ;
		for (int i = 0; i < nb; i++)
		{
			if ((int)binZ[i].size() < prm.minColPts) continue;
			qCen.push_back(xmin + (i + 0.5f) * prm.colXbin);
			qNZ.push_back(denseExtreme(binZ[i], 40.f, 15, false));	//robust near-Z, rejects flyers
		}
		if (qCen.empty()) return out;
		float gmin = *std::min_element(qNZ.begin(), qNZ.end());
		//group contiguous protruding columns (nearZ <= gmin+tol), split on a lateral gap
		std::vector<std::vector<int>> groups; std::vector<int> cur;
		for (int k = 0; k < (int)qCen.size(); k++)
		{
			bool isCone = (qNZ[k] <= gmin + prm.protrudeTol);
			if (isCone)
			{
				if (!cur.empty() && (qCen[k] - qCen[cur.back()] > prm.coneGap)) { groups.push_back(cur); cur.clear(); }
				cur.push_back(k);
			}
			else if (!cur.empty()) { groups.push_back(cur); cur.clear(); }
		}
		if (!cur.empty()) groups.push_back(cur);
		for (auto& grp : groups)
		{
			float xlo = qCen[grp.front()] - prm.colXbin, xhi = qCen[grp.back()] + prm.colXbin;
			float znear = 1e9f; for (int k : grp) znear = std::min(znear, qNZ[k]);
			pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
			for (const auto& p : crop->points)
				if (p.x >= xlo && p.x <= xhi && p.z >= znear - prm.zPad && p.z <= znear + prm.colSlab)
					c->points.push_back(p);
			if ((int)c->points.size() >= prm.minConePts)
			{
				c->width = (uint32_t)c->points.size(); c->height = 1; c->is_dense = true;
				out.push_back(c);
			}
		}
		return out;
	}

	//Depth band, FRONT-anchored: walk back from the density peak (always on the
	//dense cone, never the sparser back body) to the cone tip, hopping small gaps
	//while density >= floor (so detached front NOISE is excluded); back = front+depth.
	inline void coneZExtent(std::vector<float> z, const ChassisRoiParams& prm, float& zf, float& zb)
	{
		const float mind = 120.f;
		if (z.size() < 8) { float mn = z.empty() ? 0.f : *std::min_element(z.begin(), z.end()); zf = mn; zb = mn + mind; return; }
		float zmin = *std::min_element(z.begin(), z.end()), zmax = *std::max_element(z.begin(), z.end());
		int nb = std::max(2, (int)std::ceil((zmax - zmin) / prm.zBinw)) + 1;
		std::vector<int> h(nb, 0);
		for (float v : z) { int bi = (int)((v - zmin) / prm.zBinw); if (bi < 0) bi = 0; if (bi >= nb) bi = nb - 1; h[bi]++; }
		int pk = 0; for (int i = 1; i < nb; i++) if (h[i] > h[pk]) pk = i;
		int ghop = std::max(1, (int)std::lround(prm.zMaxGap / prm.zBinw));
		int fi = pk;
		while (true)
		{
			int nxt = -1;
			for (int k = fi - 1; k >= std::max(0, fi - ghop); k--) if (h[k] >= prm.zAbsFloor) { nxt = k; break; }
			if (nxt < 0) break; fi = nxt;
		}
		float front = zmin + fi * prm.zBinw;
		zf = front; zb = front + std::max(prm.zDepth, mind);
	}

	//Height band = cone + plate + ~yBelow under the plate. Dense Y-cluster grown
	//from cy, gap-hopping (drops gap-separated container above + wheel below),
	//capped at [cy-reachDown, cy+reachUp]. The occupancy threshold uses the peak
	//of the CONE REGION only, so the wheel's huge density can't drop the plate.
	inline void conePlateYBand(std::vector<float> yv, float cy, const ChassisRoiParams& prm, float& ylo, float& yhi)
	{
		//IGH guide is taller than the XT cone -> reach further UP so the guide top-left corner
		//(the target) is not clipped (#2 FRONT ~80mm H under-read). XT keeps yReachUp.
		const float reachUp = prm.guideDenseY ? prm.guideReachUp : prm.yReachUp;
		if ((int)yv.size() < prm.yMinAbs) { ylo = yv.empty() ? cy : *std::min_element(yv.begin(), yv.end()); yhi = yv.empty() ? cy : *std::max_element(yv.begin(), yv.end()); return; }
		float ymin = *std::min_element(yv.begin(), yv.end()), ymax = *std::max_element(yv.begin(), yv.end());
		float lo = std::floor(ymin / prm.yBinw) * prm.yBinw, hi = std::ceil(ymax / prm.yBinw) * prm.yBinw;
		int nb = std::max(1, (int)std::lround((hi - lo) / prm.yBinw)) + 1;
		std::vector<int> h(nb, 0);
		for (float v : yv) { int bi = (int)((v - lo) / prm.yBinw); if (bi < 0) bi = 0; if (bi >= nb) bi = nb - 1; h[bi]++; }
		int hmaxAll = 0; for (int x : h) hmaxAll = std::max(hmaxAll, x);
		if (hmaxAll == 0) { ylo = ymin; yhi = ymax; return; }
		int localPeak = 0;
		for (int i = 0; i < nb; i++) { float bc = lo + i * prm.yBinw; if (bc >= cy - prm.yReachDown && bc <= cy + reachUp) localPeak = std::max(localPeak, h[i]); }
		if (localPeak == 0) localPeak = hmaxAll;
		float thr = std::max((float)prm.yMinAbs, prm.yPeakFrac * localPeak);
		std::vector<bool> occ(nb, false); bool any = false;
		for (int i = 0; i < nb; i++) { occ[i] = h[i] >= thr; any = any || occ[i]; }
		if (!any) { ylo = ymin; yhi = ymax; return; }
		int ci = (int)((cy - lo) / prm.yBinw); if (ci < 0) ci = 0; if (ci >= nb) ci = nb - 1;
		if (!occ[ci]) { int best = -1; for (int i = 0; i < nb; i++) if (occ[i] && (best < 0 || std::abs(i - ci) < std::abs(best - ci))) best = i; if (best >= 0) ci = best; }
		int ghop = std::max(1, (int)std::lround(prm.yMaxGap / prm.yBinw));
		int loI = ci;
		while (true) { int nxt = -1; for (int j = loI - 1; j >= std::max(0, loI - ghop); j--) if (occ[j]) { nxt = j; break; } if (nxt < 0 || lo + nxt * prm.yBinw < cy - prm.yReachDown) break; loI = nxt; }
		int hiI = ci;
		while (true) { int nxt = -1; for (int j = hiI + 1; j <= std::min(nb - 1, hiI + ghop); j++) if (occ[j]) { nxt = j; break; } if (nxt < 0 || lo + nxt * prm.yBinw > cy + reachUp) break; hiI = nxt; }
		yhi = std::min(lo + (hiI + 1) * prm.yBinw, cy + reachUp);
		float plate = lo + loI * prm.yBinw;
		float wfloor = -1e9f;
		for (int j = loI - 1; j >= 0; j--) if (occ[j]) { wfloor = lo + (j + 1) * prm.yBinw; break; }
		ylo = std::max(std::max(plate - prm.yBelow, wfloor), cy - prm.yReachDown);
	}
}

//Extract the target cone+plate REGION from the preset crop. Returns the region
//cloud (empty if not found); conePosOut = its centroid (for the tracker / A-B).
//Feed the returned cloud as pcCone so position_detection_pca localizes it.
inline pcl::PointCloud<pcl::PointXYZ>::Ptr chassis_roi_extract_region(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& crop,
	bool targetOnRight,
	const ChassisRoiParams& prm,
	pcl::PointXYZ& conePosOut,
	const std::string& savePath = std::string(""),
	const std::string& saveTag = std::string("ROI"))
{
	using Cloud = pcl::PointCloud<pcl::PointXYZ>;
	conePosOut = pcl::PointXYZ(-10000.f, -10000.f, -10000.f);
	Cloud::Ptr empty(new Cloud);
	if (!crop || (int)crop->points.size() < prm.minConePts) return empty;

	//1. cone height (PER-CONFIG preset; runtime estimate only as a fallback) + tight
	//   candidate band (drops the low handle/rail; on CHS20 the wheel is nearer-Z).
	float cy = (prm.coneY > -9000.f) ? prm.coneY : chassis_roi_detail::coneHeightEstimate(crop, prm);
	//IGH: the GUIDE (target) is the dominant dense-Y mass; the twist-lock cone/post ~250mm
	//ABOVE it is nearer-Z, so preset-coneY / near-Z centring latches onto the WRONG (upper)
	//structure (target Y ~+60 instead of the guide top ~-170). Centre on the dominant mass.
	if (prm.guideDenseY)
		cy = chassis_roi_detail::dominantDenseY(crop, prm.guideDenseWin, prm.guideDeckGuard);
	auto buildCandBand = [&](float cyc) {
		Cloud::Ptr cb(new Cloud);
		for (const auto& p : crop->points)
			if (p.y >= cyc - prm.candYband && p.y <= cyc + prm.candYband) cb->points.push_back(p);
		return cb;
	};
	Cloud::Ptr candBand = buildCandBand(cy);

	//2. per-column near-Z candidates + side-rule selection
	auto cands = chassis_roi_detail::coneColumns(candBand, prm);
	//A GUIDE-ONLY IGH (ighs, no twist-lock cone poking up) can have its guide top well
	//BELOW the preset coneY -- e.g. REAR CHS45, guide mass ~-375 vs preset -66 -- so the
	//preset-centred band grabs only empty space above the guide and coneColumns starves.
	//Re-centre ONCE on the crop's near-Z structure height (in the now-correct trolley crop
	//the guide IS the nearest-Z mass) and retry. Preset stays authoritative wherever it
	//already forms columns, so the CHS20-wheel case (dense preset band) never triggers this.
	if (cands.empty() && prm.coneY > -9000.f && !prm.guideDenseY)
	{
		float cyEst = chassis_roi_detail::coneHeightEstimate(crop, prm);
		if (std::fabs(cyEst - cy) > prm.candYband * 0.5f)
		{
			cy = cyEst;
			candBand = buildCandBand(cy);
			cands = chassis_roi_detail::coneColumns(candBand, prm);
		}
	}
	if ((int)candBand->points.size() < prm.minConePts || cands.empty()) return empty;
	auto xmean = [](const Cloud::Ptr& c) { double s = 0; for (const auto& p : c->points) s += p.x; return c->points.empty() ? 0.0 : s / c->points.size(); };
	Cloud::Ptr target = cands[0]; double bestX = xmean(cands[0]);
	for (auto& c : cands) { double m = xmean(c); if (targetOnRight ? (m > bestX) : (m < bestX)) { bestX = m; target = c; } }

	//3a. tight lateral window around the selected cone nose
	std::vector<float> tx; tx.reserve(target->points.size());
	for (const auto& p : target->points) tx.push_back(p.x);
	float cx = chassis_roi_detail::medianOf(tx);
	Cloud::Ptr colX(new Cloud);
	for (const auto& p : crop->points)
		if (p.x >= cx - prm.xspan * 0.5f && p.x <= cx + prm.xspan * 0.5f) colX->points.push_back(p);
	if ((int)colX->points.size() < prm.minConePts) return target;

	//3b. depth band (front-anchored) from points near the cone height
	std::vector<float> zn;
	for (const auto& p : colX->points) if (std::fabs(p.y - cy) <= prm.zYband) zn.push_back(p.z);
	if ((int)zn.size() < prm.minConePts) { zn.clear(); for (const auto& p : colX->points) zn.push_back(p.z); }
	float zf, zb; chassis_roi_detail::coneZExtent(zn, prm, zf, zb);
	Cloud::Ptr colXZ(new Cloud);
	for (const auto& p : colX->points) if (p.z >= zf && p.z <= zb) colXZ->points.push_back(p);
	if ((int)colXZ->points.size() < prm.minConePts) return target;

	//3c. height band = cone + plate + below (drops wheel/container)
	std::vector<float> yv; yv.reserve(colXZ->points.size());
	for (const auto& p : colXZ->points) yv.push_back(p.y);
	float ylo, yhi; chassis_roi_detail::conePlateYBand(yv, cy, prm, ylo, yhi);
	Cloud::Ptr region(new Cloud);
	for (const auto& p : colXZ->points) if (p.y >= ylo && p.y <= yhi) region->points.push_back(p);
	if (region->points.empty()) region = target;

	region->width = (uint32_t)region->points.size(); region->height = 1; region->is_dense = true;
	double sx = 0, sy = 0, sz = 0; for (const auto& p : region->points) { sx += p.x; sy += p.y; sz += p.z; }
	int n = (int)region->points.size();
	if (n > 0) conePosOut = pcl::PointXYZ((float)(sx / n), (float)(sy / n), (float)(sz / n));

	if (!savePath.empty() && n > 0)
	{
		try { pcl::io::savePCDFileBinaryCompressed(savePath + "_" + saveTag + "_coneRegion.pcd", *region); }
		catch (...) {}
	}
	return region;
}

//Part B: prepare the extracted region for the corner detector by keeping only
//  (a) the UPPER HALF (Y >= region mid-height): the target is the cone tip + upper plate
//      corner; the lower plate/wheel/deck below pull the corner down, so drop them; AND
//  (b) the NEAR-Z FACE (thin front slab): the chassis body sits BEHIND it, so the
//      detector's densest-slice cannot latch onto the back body.
//Validated: region -> up>=midY + near-face improves dT 9->6, dH 10->4 (CS_Small/FS_Big
//height residuals vanish) with dG ~8. Falls back to the full region if a stage gets too thin.
inline pcl::PointCloud<pcl::PointXYZ>::Ptr chassis_roi_near_face(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& region, float faceDepth = 80.f,
	bool upperHalf = true, int minPts = 20)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr up(new pcl::PointCloud<pcl::PointXYZ>);
	if (!region || region->points.empty()) return up;
	if (upperHalf)
	{
		float ymin = 1e9f, ymax = -1e9f;
		for (const auto& p : region->points) { ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y); }
		float midY = 0.5f * (ymin + ymax);
		for (const auto& p : region->points) if (p.y >= midY) up->points.push_back(p);
		if ((int)up->points.size() < minPts) up = region;
		else { up->width = (uint32_t)up->points.size(); up->height = 1; up->is_dense = true; }
	}
	else up = region;

	pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<float> zs; zs.reserve(up->points.size());
	for (const auto& p : up->points) zs.push_back(p.z);
	float zNear = chassis_roi_detail::denseExtreme(zs, 40.f, 30, false);
	for (const auto& p : up->points)
		if (p.z <= zNear + faceDepth) out->points.push_back(p);
	if ((int)out->points.size() < minPts) return up;
	out->width = (uint32_t)out->points.size(); out->height = 1; out->is_dense = true;
	return out;
}

namespace chassis_roi_detail
{
	//Dense EDGE ENDPOINT: the outermost value from the low (or high) end that still has
	//>= k neighbours within `win` mm. Unlike denseExtreme (window MEAN), this returns the
	//window's outer endpoint -- the actual edge coordinate, needed for corner rescue.
	inline float denseEdge(std::vector<float> v, float win, int k, bool findMax)
	{
		if ((int)v.size() < k) return v.empty() ? 0.f : (findMax ? *std::max_element(v.begin(), v.end()) : *std::min_element(v.begin(), v.end()));
		std::sort(v.begin(), v.end());
		int n = (int)v.size();
		if (findMax)
		{
			for (int i = n - 1; i >= k - 1; i--)
				if (v[i] - v[i - k + 1] <= win) return v[i];
			return v[n - 1];
		}
		for (int i = 0; i + k - 1 < n; i++)
			if (v[i + k - 1] - v[i] <= win) return v[i];
		return v[0];
	}
}

//Statistical outlier removal (isolated-flyer filter) for the extracted region before
//targeting. Floating scatter above the cone inflates the region's max-Y, pushing the
//upper-half mid-Y up and shaving the plate; SOR drops points whose mean distance to
//their k nearest neighbours is an outlier (mean + stdMult*std), i.e. isolated points
//only -- dense structure (plate/cone/body) is untouched, so it does NOT reintroduce
//the tail-trim failure mode.
//VOXEL-HASH accelerated (~O(n)): kNN candidates come from the 27 neighbouring 40mm
//cells only; a point with <= k candidates there has no k neighbours within ~40mm and
//is isolated BY DEFINITION -> removed. Final target accuracy identical to exact SOR
//on all 47 GT frames; brute-force O(n^2) (36ms @2k, 580ms @8k) was a real-time risk.
//(validated: ~46 pts/frame removed, within-30mm 41->43, within-40mm 43->45, no regression)
inline pcl::PointCloud<pcl::PointXYZ>::Ptr chassis_roi_sor(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& in, int k = 12, float stdMult = 2.0f, float cell = 40.f)
{
	int n = in ? (int)in->points.size() : 0;
	if (n <= k + 1) return in;
	auto cellKey = [cell](const pcl::PointXYZ& p) {
		long long i = (long long)std::floor(p.x / cell);
		long long j = (long long)std::floor(p.y / cell);
		long long q = (long long)std::floor(p.z / cell);
		return (i * 73856093LL) ^ (j * 19349663LL) ^ (q * 83492791LL);
	};
	std::unordered_map<long long, std::vector<int>> grid;
	grid.reserve(n);
	for (int i = 0; i < n; i++) grid[cellKey(in->points[i])].push_back(i);

	const float BIG = 1e18f;
	std::vector<float> md(n);
	std::vector<float> d2; d2.reserve(256);
	for (int i = 0; i < n; i++)
	{
		const auto& a = in->points[i];
		d2.clear();
		long long ci = (long long)std::floor(a.x / cell), cj = (long long)std::floor(a.y / cell), cq = (long long)std::floor(a.z / cell);
		for (long long di = -1; di <= 1; di++)
			for (long long dj = -1; dj <= 1; dj++)
				for (long long dq = -1; dq <= 1; dq++)
				{
					long long key = ((ci+di) * 73856093LL) ^ ((cj+dj) * 19349663LL) ^ ((cq+dq) * 83492791LL);
					auto it = grid.find(key);
					if (it == grid.end()) continue;
					for (int j : it->second)
					{
						if (j == i) continue;
						float dx = a.x - in->points[j].x, dy = a.y - in->points[j].y, dz = a.z - in->points[j].z;
						d2.push_back(dx*dx + dy*dy + dz*dz);
					}
				}
		if ((int)d2.size() < k) { md[i] = BIG; continue; }	//no k neighbours nearby = isolated
		std::nth_element(d2.begin(), d2.begin() + k - 1, d2.end());
		double s = 0;
		for (int j = 0; j < k; j++) s += std::sqrt((double)d2[j]);
		md[i] = (float)(s / k);
	}
	double mean = 0; int nf = 0;
	for (float v : md) if (v < BIG) { mean += v; nf++; }
	if (nf < k) return in;
	mean /= nf;
	double var = 0;
	for (float v : md) if (v < BIG) var += (v - mean)*(v - mean);
	var /= nf;
	float thr = (float)(mean + stdMult * std::sqrt(var));
	pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
	out->points.reserve(n);
	for (int i = 0; i < n; i++) if (md[i] <= thr) out->points.push_back(in->points[i]);
	if ((int)out->points.size() < 30) return in;
	out->width = (uint32_t)out->points.size(); out->height = 1; out->is_dense = true;
	return out;
}

//SPARSE-FACE X-edge rescue. On a sparse near-face (<~250 pts) the shared cone detector's
//dense X stage (min_pts=30 window) cannot reach the thin target-edge column and stops up
//to ~70mm short (proven: FS_Big_CHS20 172200 dT-74). If a supported edge (>=8 pts/40mm)
//lies >40mm beyond the detected corner in the TARGET direction, move the corner to that
//edge column: x = column median, z = column's dense near-Z endpoint. Fires ONLY on sparse
//faces + clear offset + >=12-pt column, so dense/normal frames are untouched (validated:
//exactly 2/47 frames changed, both improved, zero regressions). Y is NOT set here --
//recompute it afterwards via chassis_roi_corner_height at the new (x,z).
inline bool chassis_roi_sparse_edge_rescue(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& face, bool targetOnRight, pcl::PointXYZ& pos,
	int sparseN = 250, float minOffset = 40.f, float colTol = 60.f, int minColPts = 12)
{
	if (!face || face->points.empty() || (int)face->points.size() >= sparseN) return false;
	if (pos.x == -10000.f) return false;
	std::vector<float> xs; xs.reserve(face->points.size());
	for (const auto& p : face->points) xs.push_back(p.x);
	//tor=true -> leftmost (min-X) edge; tor=false -> rightmost (max-X) edge
	float xE = chassis_roi_detail::denseEdge(xs, 40.f, 8, !targetOnRight);
	bool beyond = targetOnRight ? (xE < pos.x - minOffset) : (xE > pos.x + minOffset);
	if (!beyond) return false;
	std::vector<float> colX, colZ;
	for (const auto& p : face->points)
		if (std::fabs(p.x - xE) <= colTol) { colX.push_back(p.x); colZ.push_back(p.z); }
	if ((int)colX.size() < minColPts) return false;
	std::sort(colX.begin(), colX.end());
	pos.x = colX[colX.size() / 2];
	pos.z = chassis_roi_detail::denseEdge(colZ, 25.f, 6, false);
	return true;
}

//Target-point HEIGHT at the detected corner. pc_cone_detection sets Y = the near-Z
//slice's global MIN-Y, which on our tall cone+plate+below region lands ~100-200mm BELOW
//the real plate corner. Instead take the LOCAL median Y of the region points around the
//corner (x,z) -> the plate surface height there (validated ~8mm vs GT, cf. 139mm for
//min-Y). Optional [floorY,ceilY] sanity band (chassis cone ~1200-1800mm off ground);
//if the local median is outside it, keep the detector's Y. Returns -10000 if no points.
inline float chassis_roi_corner_height(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& region,
	float cornerX, float cornerZ, float tol = 60.f,
	float floorY = -10000.f, float ceilY = 10000.f)
{
	if (!region) return -10000.f;
	std::vector<float> ys;
	for (const auto& p : region->points)
		if (std::fabs(p.x - cornerX) <= tol && std::fabs(p.z - cornerZ) <= tol) ys.push_back(p.y);
	if (ys.empty()) return -10000.f;
	std::sort(ys.begin(), ys.end());
	float m = ys[ys.size() / 2];
	if (m < floorY || m > ceilY) return -10000.f;	//outside plausible cone band -> reject
	return m;
}

//Crop the preset world ROI and drop invalid (0,0,0) pixels. This crop is
//handed to the existing pc_*_detection_naive chain, which does the real
//target isolation (nearest-Z slab + trolley corner).
inline pcl::PointCloud<pcl::PointXYZ>::Ptr chassis_roi_crop(
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	float xMin, float xMax, float yMin, float yMax, float zMin, float zMax,
	const std::string& savePath = std::string(""),
	const std::string& saveTag = std::string("ROI"))
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
	if (!cloud || cloud->points.empty()) return out;

	out->points.reserve(8192);
	for (const auto& p : cloud->points)
	{
		if (p.x == 0.f && p.y == 0.f && p.z == 0.f) continue;
		if (p.x < xMin || p.x > xMax) continue;
		if (p.y < yMin || p.y > yMax) continue;
		if (p.z < zMin || p.z > zMax) continue;
		out->points.push_back(p);
	}
	out->width = (uint32_t)out->points.size();
	out->height = 1;
	out->is_dense = true;

	if (!savePath.empty() && !out->points.empty())
	{
		try { pcl::io::savePCDFileBinaryCompressed(savePath + "_" + saveTag + "_roiCrop.pcd", *out); }
		catch (...) {}
	}
	return out;
}

// ============================================================================
// 3D CLPS / CLPS-OK detector for the OFFLOAD cycle (XT/IGH).
// Validated by reverse-replaying 38 mount jobs (2026-07, tools/roi_offload_replay.py
// + roi_clpsok_ab.py). Facts the design rests on:
//  - While LANDED our 3D target H reads the CONTAINER corner, +22..+169mm (median ~+70)
//    ABOVE the plate; after separation H drops to the plate and is stable immediately
//    (rebound ~0). So [H <= landedH + margin] distinguishes "plate re-exposed" from
//    "chassis lifted with the container" (which keeps H at/above the loaded level).
//  - An UNGATED stability rule fires while landed in 33/40 jobs -> the PLC landed-bit
//    gate is mandatory. Re-landing happens in real jobs (134547: lift -> re-land) ->
//    the gate must be DYNAMIC (re-land releases CLPS-OK and resets).
//  - The production VA cone-only instant rule asserted CLPS-OK while LANDED in 4 jobs
//    (98 frames, partner cone seen as idx1) and NEVER fires in 18% of jobs (VA blind).
//  - Rule timing on replay (2s cadence): fires land-off +2 frames median, +5 max,
//    0 fires while landed, insensitive to stableTol 40-60 / liftMargin 20-50.
// CLPS-OK: armed after (>=1 landed phase seen) && twistlocks locked && landed OFF;
//          fire when H valid, H <= landedH+liftMargin, and H stable over stableN frames.
// CLPS   : same gate; H staying ABOVE landedH+liftMargin (plate never re-exposed =
//          chassis rising with the container) for liftN frames. Sensitive by design
//          (site: CLPS false-positive is tolerable, CLPS-OK's is NOT).
// ============================================================================
class ClpsOffload3D
{
public:
	int   stableN = 12;		//H-stable frames for CLPS-OK (~1.2s @10Hz; 3 on 2s-cadence batch)
	float stableTol = 50.f;	//max H range inside the stable window (mm)
	float liftMargin = 30.f;	//H above landedH+this = structure not settled on the chassis
	int   liftN = 15;		//frames above margin to assert CLPS (~1.5s @10Hz)

	void reset()
	{
		m_landH.clear(); m_win.clear();
		m_landedH = -10000.f; m_liftCnt = 0;
		m_everLanded = false; m_wasLanded = false; m_twlSeen = false; m_armed = false;
		m_clps = false; m_clpsOk = false;
	}

	//Feed once per OFFLOAD frame (XT/IGH). landed = PLC landed bit (visual trigger as
	//debug fallback); twlLocked = spreader twistlock locked bit (pass true when PLC
	//unavailable); H = our 3D target height, -10000 = invalid.
	void update(bool landed, bool twlLocked, float H)
	{
		if (twlLocked) m_twlSeen = true;
		bool hValid = (H > -9000.f);
		if (landed)
		{
			//(re-)landed: release both flags, reset windows, learn the loaded-phase level
			m_clpsOk = false; m_clps = false;
			m_win.clear(); m_liftCnt = 0; m_armed = false;
			m_everLanded = true; m_wasLanded = true;
			if (hValid && (int)m_landH.size() < 300)
			{
				m_landH.push_back(H);
				std::vector<float> t(m_landH);
				std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
				m_landedH = t[t.size() / 2];
			}
			return;
		}
		if (m_wasLanded)		//fresh land-off -> arm (needs a landed phase + twl seen)
		{
			m_armed = m_everLanded && m_twlSeen;
			m_win.clear(); m_liftCnt = 0; m_wasLanded = false;
		}
		if (!m_armed) return;
		if (!hValid) { m_win.clear(); return; }
		bool hasBase = (m_landedH > -9000.f);
		if (hasBase && H > m_landedH + liftMargin)
		{
			//structure still at/above the loaded level -> not settled; possible chassis lift
			m_win.clear();
			if (m_liftCnt < liftN) m_liftCnt++;
			if (m_liftCnt >= liftN) m_clps = true;
			return;
		}
		if (m_liftCnt > 0 && --m_liftCnt == 0) m_clps = false;
		//plate-level H: CLPS-OK on sustained stability (and no active CLPS)
		m_win.push_back(H);
		if ((int)m_win.size() > stableN) m_win.erase(m_win.begin());
		if ((int)m_win.size() == stableN && !m_clps)
		{
			float mn = m_win[0], mx = m_win[0];
			for (float v : m_win) { mn = std::min(mn, v); mx = std::max(mx, v); }
			if (mx - mn <= stableTol) m_clpsOk = true;	//latched until re-land
		}
	}

	bool clpsOk() const { return m_clpsOk; }
	bool clps() const { return m_clps; }
	float landedH() const { return m_landedH; }

private:
	std::vector<float> m_landH, m_win;
	float m_landedH = -10000.f;
	int m_liftCnt = 0;
	bool m_everLanded = false, m_wasLanded = false, m_twlSeen = false, m_armed = false;
	bool m_clps = false, m_clpsOk = false;
};

//Tracks the detected chassis position across frames (stationary-until-transfer).
class ChassisRoiTracker
{
public:
	enum class State { ACQUIRING, LOCKED };
	using LogFn = std::function<void(const std::string&)>;

	void reset()
	{
		m_state = State::ACQUIRING;
		m_stable = 0;
		m_miss = 0;
		m_drift = 0;
		m_movedEvents = 0;
		m_cand = pcl::PointXYZ(-10000.f, -10000.f, -10000.f);
		m_base = pcl::PointXYZ(-10000.f, -10000.f, -10000.f);
	}

	//Feed the detected cone/guide position (sentinel -10000 = no detection).
	//'freeze' = spreader landed: hold baseline, skip move checks.
	//Returns true if a usable locked baseline exists this frame.
	bool update(const pcl::PointXYZ& detected, bool freeze,
		const ChassisRoiParams& prm, LogFn log = LogFn())
	{
		auto logMsg = [&](const std::string& m) { if (log) log(m); };
		bool valid = (detected.x != -10000.f && detected.y != -10000.f && detected.z != -10000.f);

		if (freeze && m_state == State::LOCKED)
		{
			logMsg("[ROI] frozen (spreader landed), holding baseline");
			return true;
		}

		if (m_state == State::ACQUIRING)
		{
			if (!valid) { m_stable = 0; return false; }

			bool agrees = (m_stable > 0)
				&& std::fabs(detected.x - m_cand.x) <= prm.lockTol
				&& std::fabs(detected.z - m_cand.z) <= prm.lockTol;
			m_stable = agrees ? m_stable + 1 : 1;
			m_cand = detected;

			if (m_stable >= prm.lockK)
			{
				m_state = State::LOCKED;
				m_base = detected;
				m_miss = 0;
				m_drift = 0;
				logMsg("[ROI] LOCKED chassis at (T:" + std::to_string((int)m_base.x)
					+ " H:" + std::to_string((int)m_base.y)
					+ " G:" + std::to_string((int)m_base.z) + ")");
				return true;
			}
			logMsg("[ROI] acquiring " + std::to_string(m_stable) + "/" + std::to_string(prm.lockK)
				+ " at (T:" + std::to_string((int)detected.x) + " G:" + std::to_string((int)detected.z) + ")");
			return false;
		}

		//LOCKED
		if (!valid)
		{
			m_miss++;
			if (m_miss > prm.maxMiss)
			{
				logMsg("[ROI] UNLOCKED (chassis lost) -> re-acquire");
				m_state = State::ACQUIRING;
				m_stable = 0; m_miss = 0; m_drift = 0;
			}
			return false;	//no fresh detection this frame
		}
		m_miss = 0;

		float dx = detected.x - m_base.x;
		float dz = detected.z - m_base.z;
		if (std::fabs(dx) > prm.moveTol || std::fabs(dz) > prm.moveTol)
		{
			m_drift++;
			logMsg("[ROI] drift " + std::to_string(m_drift) + "/" + std::to_string(prm.moveConfirm)
				+ " dT:" + std::to_string((int)dx) + " dG:" + std::to_string((int)dz));
			if (m_drift >= prm.moveConfirm)
			{
				m_movedEvents++;
				logMsg("[ROI] CHASSIS_MOVED dT:" + std::to_string((int)dx)
					+ " dG:" + std::to_string((int)dz) + " -> re-acquire");
				m_state = State::ACQUIRING;
				m_stable = 1;
				m_cand = detected;
				m_drift = 0;
			}
			return false;	//don't trust this frame's position while drifting
		}
		m_drift = 0;
		return true;
	}

	bool locked() const { return m_state == State::LOCKED; }
	std::string stateString() const { return (m_state == State::LOCKED) ? "LOCK" : "ACQ"; }
	pcl::PointXYZ baseline() const { return m_base; }
	int movedEvents() const { return m_movedEvents; }

private:
	State m_state = State::ACQUIRING;
	int m_stable = 0;
	int m_miss = 0;
	int m_drift = 0;
	int m_movedEvents = 0;

	pcl::PointXYZ m_cand{ -10000.f, -10000.f, -10000.f };	//acquire candidate
	pcl::PointXYZ m_base{ -10000.f, -10000.f, -10000.f };	//locked baseline
};
