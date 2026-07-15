# Cross-run eval: ROI-detected cone (new run) vs original VA run true cone, per frame.
param(
  [string]$VaRoot  = "K:\TUAS\South_Endfeed\ActualJob\XT_Batch\FRONT\Batch_result",
  [string]$RoiRoot = "K:\TUAS\South_Endfeed\ActualJob\XT_Batch\FRONT\Batch_result_ROI2"
)

function Load-Cones($root, $coneCols) {
  $map = @{}
  foreach ($dir in (Get-ChildItem $root -Directory)) {
    if ($dir.Name -match "TMINI" -or $dir.Name -eq "Batch_result") { continue }
    $res = Join-Path $dir.FullName ($dir.Name + "_res.txt")
    if (-not (Test-Path $res)) { continue }
    $lines = Get-Content $res
    $hdr = $lines[0] -split ";"
    $ixF = 0
    $ci = @{}; foreach ($c in $coneCols) { $ci[$c] = [array]::IndexOf($hdr, $c) }
    foreach ($ln in ($lines | Select-Object -Skip 1)) {
      $a = $ln -split ";"
      if ($a.Count -le ($ci.Values | Measure-Object -Maximum).Maximum) { continue }
      $key = $dir.Name + "|" + $a[$ixF]
      $row = @{}
      foreach ($c in $coneCols) { $row[$c] = $a[$ci[$c]] }
      $map[$key] = $row
    }
  }
  return $map
}

$va  = Load-Cones $VaRoot  @("cone_x","cone_y","cone_z")
$roi = Load-Cones $RoiRoot @("ROI_T","ROI_H","ROI_G","ROI_State","AB_Valid","AB_dT","AB_dH","AB_dG")

$rows = New-Object System.Collections.Generic.List[object]
foreach ($k in $roi.Keys) {
  if (-not $va.ContainsKey($k)) { continue }
  $vx=[double]$va[$k].cone_x; $rx=[double]$roi[$k].ROI_T
  if ($vx -eq -10000 -or $rx -eq -10000) { continue }
  $job = ($k -split "\|")[0]
  $short = (($job -split "_") | Select-Object -Skip 4 -First 5) -join "_"
  $rows.Add([pscustomobject]@{
    Job=$short
    dT=[math]::Abs($rx - $vx)
    dH=[math]::Abs([double]$roi[$k].ROI_H - [double]$va[$k].cone_y)
    dG=[math]::Abs([double]$roi[$k].ROI_G - [double]$va[$k].cone_z)
    RT=$rx; VT=$vx
  })
}

Write-Output ("Matched frames (both cones valid): " + $rows.Count)
if ($rows.Count -gt 0) {
  foreach ($ax in @("dT","dH","dG")) {
    $v = $rows.$ax | Sort-Object
    $med=$v[[int]($v.Count/2)]; $p95=$v[[int]([math]::Min($v.Count-1,$v.Count*0.95))]
    Write-Output ("|{0}|  median {1,6:F0}  p95 {2,6:F0}  max {3,6:F0}  mm" -f $ax,$med,$p95,$v[-1])
  }
  $good = ($rows | Where-Object { $_.dT -le 100 }).Count
  Write-Output ("Frames with |dT|<=100mm: {0} / {1} ({2:P0})" -f $good,$rows.Count,($good/$rows.Count))
  Write-Output "--- per job: |dT| median / agreement ---"
  $rows | Group-Object Job | Sort-Object Name | ForEach-Object {
    $v = $_.Group.dT | Sort-Object
    $g = ($_.Group | Where-Object { $_.dT -le 100 }).Count
    "{0,-28} n:{1,4} |dT|med:{2,6:F0}  <=100mm:{3,3:P0}  RT~{4,6:F0} VT~{5,6:F0}" -f `
      $_.Name,$v.Count,$v[[int]($v.Count/2)],($g/$_.Group.Count),($_.Group.RT|Measure-Object -Average).Average,($_.Group.VT|Measure-Object -Average).Average
  }
}
