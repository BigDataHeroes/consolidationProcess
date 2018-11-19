#!/bin/bash

source properties.sh

python3.7 consolidation_v2.py $distrito_barrio $datasetIndicadores $opinionesTweeter $EvolucionPreciosAlquiler $transporte_agregado $IntervencionesPMun_unix_utf8 $barrios_distritos_NO2 $barrios_distritos_m25 $barrios_edge_file $colegios $datos_consolidados_fin
