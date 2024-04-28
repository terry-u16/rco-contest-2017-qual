param(
    [Parameter(mandatory)]
    [int]
    $seed
)

$in = ".\data\inA\{0:0000}.txt" -f $seed
$env:DURATION_MUL = "1.5"
Get-Content $in | cargo run --release --bin rco-contest-2017-qual-a > .\out.txt
