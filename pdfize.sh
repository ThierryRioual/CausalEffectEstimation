
CONVERTPDF="jupyter nbconvert --embed-images --template=widelab --to webpdf --output-dir=outputs"
CONVERTPDFNOCODE="$CONVERTPDF --no-input --no-prompt --output={notebook_name}-nocode"

$CONVERTPDF 1-Potential\ Outcome\ and\ Randomized\ Control\ Trial/Lab1pyAgrum.ipynb
$CONVERTPDFNOCODE 1-Potential\ Outcome\ and\ Randomized\ Control\ Trial/Lab1pyAgrum.ipynb