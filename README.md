# Overlappogram
Can do Low Fip High Fip Response functions.
make_spectral= false to use LOWFIP, highsfip response functions

in TOML file, instead of response ='', use lowfip='' and highfip=''

## Example 
<br/>
[paths]<br/>
overlappogram = "flight.fits"<br/>
weights = "weights.fits"<br/>
lowfip='lowfip_response.fits'<br/>
highfip='highfip_response.fits'<br/>
...<br/>
[output]<br/>
...<br/>
make_spectral=false<br/>
...<br/>
