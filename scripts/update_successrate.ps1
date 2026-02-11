param(
    [string]$OutDir = "tifa_dataset",
    [string]$LogDir = "logs",
    [string]$Output = "SUCCESSRATE.MD"
)

$ErrorActionPreference = "Stop"

$pythonScript = @'
import sqlite3
import sys
import json
from pathlib import Path

out_dir = Path(sys.argv[1])
db_path = out_dir / "images.db"
if not db_path.exists():
    print(json.dumps({"error": f"missing db at {db_path}"}))
    sys.exit(0)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

cur = conn.execute("SELECT COUNT(*) AS c FROM images")
total = cur.fetchone()["c"]
cur = conn.execute("SELECT COUNT(*) AS c FROM images WHERE category != 'discard'")
kept = cur.fetchone()["c"]
cur = conn.execute("SELECT COUNT(*) AS c FROM images WHERE category = 'discard'")
discard = cur.fetchone()["c"]

cur = conn.execute(
    "SELECT model_name, COUNT(*) AS c "
    "FROM images WHERE category='discard' "
    "GROUP BY model_name ORDER BY c DESC"
)
discard_reasons = [{"reason": row["model_name"], "count": row["c"]} for row in cur.fetchall()]

cur = conn.execute(
    "SELECT category, COUNT(*) AS c "
    "FROM images GROUP BY category ORDER BY c DESC"
)
categories = [{"category": row["category"], "count": row["c"]} for row in cur.fetchall()]

conn.close()
print(json.dumps({
    "total": total,
    "kept": kept,
    "discard": discard,
    "discard_reasons": discard_reasons,
    "categories": categories
}))
'@

$metricsJson = $pythonScript | python - $OutDir
$metrics = $metricsJson | ConvertFrom-Json

$latestLog = Get-ChildItem -Path $LogDir -Filter *.log -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

$logMetrics = [ordered]@{
    latest_log = $null
    download_failed = 0
    classification_failed = 0
    discarded_invalid = 0
    decode_skipped = 0
    rate_limited = 0
    search_failed = 0
    download_reasons = @()
}

if ($latestLog) {
    $logMetrics.latest_log = $latestLog.FullName
    $lines = Get-Content -Path $latestLog.FullName
    $logMetrics.download_failed = ($lines | Where-Object { $_ -match 'download failed' }).Count
    $logMetrics.classification_failed = ($lines | Where-Object { $_ -match 'classification failed' }).Count
    $logMetrics.discarded_invalid = ($lines | Where-Object { $_ -match 'discarded invalid image' }).Count
    $logMetrics.decode_skipped = ($lines | Where-Object { $_ -match 'image decode skipped' }).Count
    $logMetrics.rate_limited = ($lines | Where-Object { $_ -match 'status 429' }).Count
    $logMetrics.search_failed = ($lines | Where-Object { $_ -match 'search failed' }).Count

    $downloadReasons = $lines |
        Where-Object { $_ -match 'download failed' } |
        ForEach-Object { if ($_ -match 'error=(.+)$') { $Matches[1] } } |
        Group-Object | Sort-Object Count -Descending

    $logMetrics.download_reasons = $downloadReasons |
        ForEach-Object { "{0}={1}" -f $_.Name, $_.Count }
}

$today = Get-Date -Format "yyyy-MM-dd"

$content = New-Object System.Collections.Generic.List[string]
$content.Add("# SUCCESSRATE.MD")
$content.Add("")
$content.Add("Last updated: $today")
$content.Add("")
$content.Add("## Baseline metrics (cumulative, from $OutDir\\images.db)")

if ($metrics.error) {
    $content.Add("- ERROR: $($metrics.error)")
} else {
    $total = [int]$metrics.total
    $kept = [int]$metrics.kept
    $discard = [int]$metrics.discard
    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $keptRate = if ($total -gt 0) { ((($kept / $total) * 100).ToString("0.0", $invariant) + "%") } else { "0.0%" }
    $discardRate = if ($total -gt 0) { ((($discard / $total) * 100).ToString("0.0", $invariant) + "%") } else { "0.0%" }

    $content.Add("- Total records: $total")
    $content.Add("- Kept: $kept ($keptRate)")
    $content.Add("- Discarded: $discard ($discardRate)")
    $content.Add("- Top discard reasons (model_name):")
    if ($metrics.discard_reasons.Count -eq 0) {
        $content.Add("  - (none)")
    } else {
        foreach ($reason in $metrics.discard_reasons | Select-Object -First 10) {
            $content.Add("  - $($reason.reason)=$($reason.count)")
        }
    }

    $content.Add("- Category counts:")
    foreach ($category in $metrics.categories) {
        $content.Add("  - $($category.category)=$($category.count)")
    }
}

$content.Add("")
$content.Add("## Latest run log metrics")
if (-not $logMetrics.latest_log) {
    $content.Add("- No log files found in $LogDir")
} else {
    $content.Add("- latest_log: $($logMetrics.latest_log)")
    $content.Add("- download_failed: $($logMetrics.download_failed)")
    $content.Add("- classification_failed: $($logMetrics.classification_failed)")
    $content.Add("- discarded_invalid: $($logMetrics.discarded_invalid)")
    $content.Add("- image_decode_skipped: $($logMetrics.decode_skipped)")
    $content.Add("- rate_limited: $($logMetrics.rate_limited)")
    $content.Add("- search_failed: $($logMetrics.search_failed)")
    $content.Add("- download_reasons:")
    if ($logMetrics.download_reasons.Count -eq 0) {
        $content.Add("  - (none)")
    } else {
        foreach ($item in $logMetrics.download_reasons) {
            $content.Add("  - $item")
        }
    }
}

Set-Content -Path $Output -Value $content -Encoding UTF8
