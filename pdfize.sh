CONVERTPDF="jupyter nbconvert --embed-images --template=widelab --to webpdf --output-dir=outputs"
CONVERTPDFNOCODE="$CONVERTPDF --no-input --no-prompt --output={notebook_name}-nocode"

$CONVERTPDF notebooks/pyAgrumLab1a.ipynb  
$CONVERTPDF notebooks/pyAgrumLab1b.ipynb  
$CONVERTPDF notebooks/pyAgrumLab2.ipynb   
$CONVERTPDF notebooks/pyAgrumLab4.ipynb   
$CONVERTPDF notebooks/pyAgrumLab5.ipynb

$CONVERTPDFNOCODE notebooks/pyAgrumLab1a.ipynb  
$CONVERTPDFNOCODE notebooks/pyAgrumLab1b.ipynb  
$CONVERTPDFNOCODE notebooks/pyAgrumLab2.ipynb   
$CONVERTPDFNOCODE notebooks/pyAgrumLab4.ipynb   
$CONVERTPDFNOCODE notebooks/pyAgrumLab5.ipynb
