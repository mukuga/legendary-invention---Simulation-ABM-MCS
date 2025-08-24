# ============================================
# 🔁 Jalankan semua sesi: region × skenario
# PowerShell - versi penuh
# powershell -ExecutionPolicy Bypass -File .\run-all.ps1
# ============================================

# 1️⃣ Daftar wilayah & skenario
$regions = @("indonesia", "united_states", "eurozone")
$scenarios = @(
    "S0_baseline", "S1_depeg_stablecoin", "S2_panic_selling_defi",
    "S3_twin_shock", "S4_twin_shock_cex_intervention",
    "S5_twin_shock_imf_bailout", "S6_worst_case_no_intervention"
)
$coin = "USDC"

# 2️⃣ Loop: Build config → Run sim → Visualisasi
foreach ($region in $regions) {
    foreach ($scenario in $scenarios) {
        Write-Host "`n=== [$region][$scenario] ==="

        # ▪ Buat YAML config
        $cfg = "config_${scenario}_${coin}_${region}.yml"
        python param_loader_2.py `
            --scenario $scenario `
            --region $region `
            --coin $coin `
            --out $cfg

        # ▪ Jalankan simulasi → simpan ke folder per region+skenario
        $outdir = "results/$region/$scenario"
        New-Item -ItemType Directory -Force -Path $outdir | Out-Null
        python run.py `
            --config $cfg `
            --outdir $outdir

        # ▪ Visualisasi hasil ke folder figures
        $figdir = "figures/$region/$scenario"
        New-Item -ItemType Directory -Force -Path $figdir | Out-Null
        python Visualisasi.py `
            --data-dir $outdir `
            --output-dir $figdir
    }
}
Write-Host "`n✅ Semua simulasi & visualisasi selesai!"



