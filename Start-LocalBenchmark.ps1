Write-Host "[Compile]"
cargo build --release --bin rco-contest-2017-qual-a
Move-Item ../target/release/rco-contest-2017-qual-a.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
dotnet marathon run-local
#./relative_score.exe -d ./data/results -o min