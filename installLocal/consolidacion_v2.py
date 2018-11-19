# -*- coding: utf-8 -*-


import nltk
import pandas as pd
import numpy as np
from hdfs3 import HDFileSystem
import sys
import unidecode
nltk.download('stopwords')
from nltk.corpus import stopwords


distrito_barrio=sys.argv[1]  #distrito_barrio.csv
datasetIndicadores=sys.argv[2] # datasetIndicadores.csv
opinionesTweeter =sys.argv[3] #opinionesTweeter20181024.csv
EvolucionPreciosAlquiler=sys.argv[4] #EvolucionPreciosAlquiler
transporte_agregado=sys.argv[5] #transporte_agregado.csv
IntervencionesPMun_unix_utf8=sys.argv[6] #IntervencionesPMun_unix_utf8.csv
barrios_distritos_NO2=sys.argv[7] #barrios_distritos_NO2.csv
barrios_distritos_m25=sys.argv[8] #barrios_distritos_m25.csv
barrios_edge_file=sys.argv[9] #r'/content/drive/My Drive/Eduardo/data/Barrios/BARRIOS.geojson'
colegios=sys.argv[10]   #"/content/drive/My Drive/Jaime/colegios.csv"
datos_consolidados_fin=sys.argv[11] # output

hdfs = HDFileSystem(host='bdhKC', port=9000)


# Cargo lookup de barrios
with hdfs.open(distrito_barrio) as f:
  datos_distrito_barrio = pd.read_csv(f)

# limpio a ascii
stop_words = []
datos_distrito_barrio['DISTRITOASCII'] = datos_distrito_barrio['DISTRITO'].apply(unidecode.unidecode)
datos_distrito_barrio['DISTRITOASCII'] = datos_distrito_barrio['DISTRITOASCII'].astype('str').str.lower()

datos_distrito_barrio['BARRIOASCII'] = datos_distrito_barrio['BARRIO'].apply(unidecode.unidecode)
datos_distrito_barrio['BARRIOASCII'] = datos_distrito_barrio['BARRIOASCII'].astype('str').str.lower()

# Quito stop words
from nltk.corpus import stopwords
stop = stopwords.words('spanish')
datos_distrito_barrio['DISTRITOASCII'] = datos_distrito_barrio['DISTRITOASCII'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
datos_distrito_barrio['BARRIOASCII'] = datos_distrito_barrio['BARRIOASCII'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Quito duplicados
datos_distrito_barrio = datos_distrito_barrio.drop_duplicates()

"""* Vamos a utilizar el CODBAR como dimensión de unión. 
* Las dimensiones del dataset se ponen en minúsculas, las de los barrios/distritos en mayúsculas (CODDIST,DISTRITO,CODBAR,BARRIO,DISTRITOASCII,BARRIOASCII)
* Si el dataset a consolidar ya la tiene el merge es sencillo. 
* Si no, intentamos cruzar con el campo ASCII que es un nombre nominal (sin acentos, stop words, etc...)
"""

# Estos estan duplicados, los barrios han aparecido y he duplicado el codigo de barrio pero con nombre distinto. Viene del fichero distrito_barrio.csv

datos_distrito_barrio[datos_distrito_barrio.duplicated('CODBAR')]

"""# Estudio Calidad de Vida - Agregados"""

with hdfs.open(datasetIndicadores) as f:
  datos_calidad_vida = pd.read_csv(datasetIndicadores, sep=';')

datos_calidad_vida = datos_calidad_vida.replace(' - ',np.NaN)

datos_calidad_vida.columns = ['lugar',
'superficie',
'densidad',
'sexo',
'hombres',
'mujeres',
'edad_media',
'de_0_a_14_anios',
'de_15_a_29',
'de_30_a_44',
'de_45_a_64',
'de_65_a_79',
'de_80_y',
'de_65_y',
'poblacion_en_etapa_educativa_de_3_a_16_anios_-16_no_incluidos',
'personas_con_nacionalidad_espaniola',
'personas_con_nacionalidad_espaniola_hombres',
'personas_con_nacionalidad_espaniola_mujeres',
'personas_con_nacionalidad_extranjera',
'personas_con_nacionalidad_extranjera_hombres',
'personas_con_nacionalidad_extranjera_mujeres',
'n_total_de_hogares',
'tamanio_medio_del_hogar',
'hogares_con_una_mujer_sola_>65_anios',
'hogares_con_un_hombre_solo_>65_anios',
'hogares_monoparentales_una_mujer_adulta_con_uno_o_mas_menores',
'hogares_monoparentales_un_hombre_adulto_con_uno_o_mas_menores',
'tasa_bruta_de_natalidad_2016_‰',
'tasa_de_crecimiento_demografico_2016_‰',
'esperanza_de_vida_al_nacer_mujeres_mortalidad_del_periodo_2009-2012',
'esperanza_de_vida_al_nacer_hombres_mortalidad_del_periodo_2009-2012',
'esperanza_de_vida_al_nacer_mujeres',
'esperanza_de_vida_al_nacer_hombres',
'esperanza_de_vida_>_65_anios_mujeres',
'esperanza_de_vida_>_65_anios_hombres',
'renta_neta_media_anual_de_los_hogares_urban_audit__2014',
'renta_disponible_bruta_per_capita_2014_a',
'crecimientodecrecimiento_renta_bruta_per_capita_2014_a_-2008',
'pension_media_mensual_del_distrito_hombres_2015',
'pension_media_mensual_del_distrito_mujeres_2015',
'numero_de_parados_ciudad_de_madrid_epa_3_trimestre_2017',
'hombres',
'mujeres1',
'paro_registrado_n_de_personas_registradas_en_sepe_en_agosto_2017',
'parados_de_larga_duracion_agosto_2017',
'hombres1',
'mujeres',
'parados_que_si_perciben_prestaciones_agosto_2017',
'hombres2',
'mujeres_1',
'parados_que_no_perciben_prestaciones_agosto_2017',
'hombres3',
'mujeres_2',
'151_poblacion_en_etapas_educativas_01012017',
'0_a_2_anios',
'infantil_3_a_5_anios',
'primaria_6_a_11_anios',
'secundaria_12_a_15_anios',
'etapas_educativas_total_ninios',
'0_a_2_anios_ninios',
'infantil_3_a_5_anios_ninios',
'primaria_6_a_11_anios_ninios',
'secundaria_12_a_15_anios_ninios',
'etapas_educativas_total_ninias',
'0_a_2_anios_ninias',
'infantil_3_a_5_anios_ninias',
'primaria_6_a_11_anios_ninias',
'secundaria_12_a_15_anios_ninias',
'152_escolarizacion_de_alumnos_por_tipo_de_centro_anio_escolar_201516',
'en_centros_privados_concertados',
'en_centros_privados_sin_concierto',
'en_centros_publicos',
'total_alumnos_extranjeros',
'en_centros_privados_concertados1',
'en_centros_privados_sin_concierto1',
'en_centros_publicos1',
'total_alumnos_con_necesidades_de_apoyo_educativo',
'en_centros_privados_concertados2',
'en_centros_privados_sin_concierto2',
'en_centros_publicos2',
'no_sabe_leer_ni_escribir_o_sin_estudios',
'primaria_incompleta',
'bachiller_elemental_graduado_escolar_esoformacion_profesional_1_grado',
'formacion_profesional_2_grado_bachiller_superior_o_bup',
'titulados_medios_diplomados_arquitecto_o_ingeniero_tecnico',
'estudios_superiores_licenciado_arquitecto_o_ingeniero_superior_estudios_superiores_no_universitarios_doctorado_estudios_postgraduados',
'nivel_de_estudios_desconocido_y_no_consta',
'casos_trabajados_por_el_programa_de_absentismo_municipal_curso_escolar_2015-2016',
'casos_trabajados_hombres',
'casos_trabajados_mujeres',
'calidad_de_vida_relacionada_con_la_salud_cvrs',
'calidad_de_vida_relacionada_con_la_salud_cvrs_hombres',
'calidad_de_vida_relacionada_con_la_salud_cvrs_mujeres',
'n_de_personas_con_grado_de_discapacidad_reconocido',
'hombres_discapacidad',
'mujeres_discapacidad',
'mujeres_por_cada_100_hombres_feminizacion_de_la_discapacidad_reconocida',
'satisfaccion_de_vivir_en_su_barrio',
'calidad_de_vida_actual_en_su_barrio',
'convivencia_vecinal',
'espacios_verdes',
'parques_infantiles',
'centros_culturales',
'organizacion_de_fiestas_y_eventos_populares',
'instalaciones_deportivas',
'servicios_sociales_municipales',
'percepcion_de_seguridad_en_madrid',
'percepcion_de_seguridad_en_el_barrio_por_el_dia',
'percepcion_de_seguridad_en_el_barrio_por_la_noche',
'personas_orinando_en_la_calle',
'residuos_por_excremento_de_animales_domesticos',
'molestias_por_ruido',
'pareja_o_matrimonio_con_hijos',
'hogar_formado_por_madre_con_hijos_a_cargo',
'nacido_en_espania',
'nacido_en_un_pais_fuera_de_la_union_europea',
'dificultad_para_llegar_a_fin_de_mes_respuesta_mucha_dificultad_dificultad',
'imposibilidad_de_hacer_frente_a_un_gasto_imprevisto_de_650',
'dificultad_para_hacer_frente_a_los_gastos_derivados_del_suministro_energetico',
'entrevistados_que_han_acudido_a_servicios_sociales_municipales_para_solicitar_atencion_social',
'personas_atendidas_en_la_unidad_de_primera_atencion_en_centros_de_servicios_sociales',
'solicitudes_tramitadas_de_renta_minima_de_insercion',
'perceptores_de_prestacion_de_la_renta_minima_de_insercion',
'beneficiarios_de_prestaciones_sociales_de_caracter_economico',
'total_prestaciones_sociales_de_caracter_economico_€',
'personas_con_servicio_de_ayuda_a_domicilio_modalidad_auxiliar_de_hogar',
'personas_con_servicio_de_ayuda_a_domicilio_modalidad_auxiliar_de_hogar_hombres',
'personas_con_servicio_de_ayuda_a_domicilio_modalidad_auxiliar_de_hogar_mujeres',
'personas_socias_de_los_centros_municipales_de_mayores',
'personas_socias_de_los_centros_municipales_de_mayores_hombres',
'personas_socias_de_los_centros_municipales_de_mayores_mujeres',
'familias_atendidas_por_el_equipo_de_trabajo_con_menores_y_familia_etmf',
'demandas_de_intervencion_en_los_centros_de_atencion_a_la_infancia_cai',
'menores_atendidos_por_el_servicio_de_ayuda_a_domicilio_a_menores_y_familias_sad',
'n_de_edificios',
'n_de_viviendas',
'n_total_de_viviendas_censo_edificios_y_viviendas_2011',
'viviendas_anteriores_a_1980',
'viviendas_estado_ruinoso',
'viviendas_estado_malo',
'viviendas_estado_deficiente',
'viviendas_estado_bueno',
'viviendas_no_consta',
'valor_catastral_medio_de_los_bienes_inmuebles_personas_fisicas_2016',
'valor_catastral_medio_de_los_bienes_inmuebles_personas_juridicas_2016',
'superficie_media_de_la_vivienda_m2_en_transaccion_2016',
'duracion_media_del_credito_meses_en_transaccion_de_vivienda_2016',
'n_total_de_viviendas_familiares_censo_edificios_y_viviendas_2011',
'n_total_de_viviendas_familiares_censo_edificios_y_viviendas_2011_principal',
'n_total_de_viviendas_familiares_censo_edificios_y_viviendas_2011_secundaria',
'n_total_de_viviendas_familiares_censo_edificios_y_viviendas_2011_desocupada',
'n_total_de_viviendas_familiares_censo_edificios_y_viviendas_2011_otro_tipo',
'lanzamientos_judiciales_practicados_2013',
'como_consecuencia_de_ejecuciones_hipotecarias',
'como_consecuencia_de_la_ley_de_arrendamientos_urbanos',
'lanzamientos_judiciales_practicados_2016',
'como_consecuencia_de_ejecuciones_hipotecarias1',
'como_consecuencia_de_la_ley_de_arrendamientos_urbanos1',
'particulas_suspension_pm25_µgrm3_-valor_limite_25_µgrm3',
'particulas_suspension_pm10_µgrm3_-valor_limite_20_µgrm3',
'dioxido_de_azufre_-_so2_µgrm3_-valor_limite_125_µgrm3',
'monoxido_de_carbono_-_co2_mgm3_-valor_limite_10_µgrm3',
'ozono_-_o3_µgrm3_-valor_limite_120_µgrm3',
'dioxido_de_nitrogeno_-_no2_µgrm3_-valor_limite_40_µgrm3',
'temperatura_media_anual_c',
'kghabitanteanio',
'kghabitantedia',
'relacionadas_con_las_personas',
'relacionadas_con_la_tenencia_de_armas',
'relacionadas_con_el_patrimonio',
'relacionadas_con_la_tenencia_y_consumo_de_drogas',
'expedientes_instruidos_por_agentes_tutores',
'atestadospartes_de_accidentes_de_trafico_confeccionados',
'detenidos_e_investigados_total',
'detenidos_e_investigados_lesiones',
'detenidos_e_investigados_violencia_domestica',
'detenidos_e_investigados_malos_tratos_a_menores',
'detenidos_e_investigados_abusos_y_agresiones_sexuales',
'detenidos_e_investigados_hurtos',
'detenidos_e_investigados_robo_con_fuerza_o_violencia_e_intimidacion',
'detenidos_e_investigados_contra_la_salud_publica',
'detenidos_e_investigados_contra_la_seguridad_vial',
'censo_electoral',
'votantes_abstencion',
'votantes_votos_blancos',
'votantes_votos_a_candidaturas',
'votantes_pp',
'votantes_psoe',
'votantes_ahora_madrid',
'votantes_ciudadanos',
'centros_de_servicios_sociales',
'centros_municipales_de_mayores',
'centros_de_dia_de_alzheimer_y_fisicos',
'apartamentos_municipales_para_mayores',
'residencias_de_mayores',
'centros_de_atencion_a_la_infancia_cai',
'n_de_bibliotecas_municipales',
'n_de_bibliotecas_comunidad_madrid',
'n_de_centros_culturales',
'centros_deportivos_municipales_totales',
'centros_deportivos_municipales_gestion_directa',
'centros_deportivos_municipales_gestion_indirecta',
'superficie_deportiva_m2',
'superficie_deportiva_m2_en_cdm_gestion_directa',
'superficie_deportiva_m2_en_cdm_gestion_indirecta',
'instalaciones_deportivas_basicas_2016',
'instalaciones_deportivas_basicas_2016_acceso_controlado',
'm2_de_superficie_en_idb_acceso_controlado',
'acceso_libre',
'm2_de_superficie_en_idb_acceso_libre',
'campos_de_futbol',
'pista_de_atletismo',
'piscinas_cubiertas',
'piscinas_de_verano',
'zonas_verdes_ha',
'relacion_de_zona_verde_m2_habitante_deseables_15m2hab_minimo_10m2hab',
'escuelas_infantiles_municipales',
'escuelas_infantiles_publicas',
'escuelas_infantiles_privadas',
'colegios_publicos_infantil_y_primaria__3_centros_con_secundaria',
'institutos_publicos_de_educacion_secundaria',
'colegios_privados_inf_pri',
'colegios_privados_inf_pri_sec',
'mercados_municipales',
'n_total_de_asociaciones_2016',
'n_de_asociaciones_de_caracter_social',
'n_de_asociaciones_de_vecinos']

# calculo columna de cruce. "Lugar" => "BARRIOASCII"

datos_calidad_vida['BARRIOASCII'] = datos_calidad_vida['lugar'].apply(unidecode.unidecode)
datos_calidad_vida['BARRIOASCII'] = datos_calidad_vida['BARRIOASCII'].astype('str').str.lower()

stop = stopwords.words('spanish')
datos_calidad_vida['BARRIOASCII'] = datos_calidad_vida['BARRIOASCII'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
datos_calidad_vida[['lugar','BARRIOASCII']].head()

datos_calidad_vida = pd.merge(datos_calidad_vida, datos_distrito_barrio, how='left', on='BARRIOASCII')


# Marco los barrios para tenerlos localizados 
datos_calidad_vida['BARRIO'] = datos_calidad_vida['BARRIO'].fillna('BORRAR')

# Como están ordenados jerarquicamente, simplemente los estiro hacia abajo, y los barrios heredan los valores de los distritos. 
datos_calidad_vida = datos_calidad_vida.fillna(method='ffill')

# Y ahora borro los "BORRAR"
datos_calidad_vida = datos_calidad_vida[datos_calidad_vida['BARRIO'] != 'BORRAR']

datos_calidad_vida = datos_calidad_vida.drop(['sexo','personas_con_nacionalidad_espaniola_hombres',
'personas_con_nacionalidad_espaniola_mujeres','personas_con_nacionalidad_extranjera_hombres',
'personas_con_nacionalidad_extranjera_mujeres','hombres','mujeres1','hombres1','mujeres','hombres2',
'mujeres_1','hombres3', 'mujeres_2', 'superficie_deportiva_m2', 'numero_de_parados_ciudad_de_madrid_epa_3_trimestre_2017'
], axis=1)



"""# AirBNB - Agregado"""
with hdfs.open(airbnb_aggregate) as f:
  datos_airbnb = pd.read_csv(f)
datos_airbnb = datos_airbnb.drop(['Unnamed: 0'], axis = 1)
datos_airbnb.columns = ['DISTRITO','CODDIST','BARRIO','CODBAR','airbnb_precio_noche','airbnb_precio_semana','airbnb_precio_mes','airbnb_conteo']


"""# Twitter"""
with hdfs.open(opinionesTweeter) as f:
  datos_twitter = pd.read_csv(f)

datos_twitter = datos_twitter.groupby(['Distrito']).agg('mean').reset_index()

# excepciones
datos_twitter = datos_twitter.replace('CiudadLineal', 'Ciudad lineal')
datos_twitter = datos_twitter.replace('PuenteVallecas', 'Puente Vallecas')
datos_twitter = datos_twitter.replace('VillaVallecas', 'Villa Vallecas')
datos_twitter = datos_twitter.replace('Canillejas', 'San Blas - Canillejas')
datos_twitter = datos_twitter.replace('Moncloa', 'Moncloa - Aravaca')
datos_twitter = datos_twitter.replace('Fuencarral', 'Fuencarral - El Pardo')
datos_twitter.columns=['Distrito','twit_op+','twit_op-']

# cruzo el distrito y saco su codigo.  "Distrito" = "DISTRITOASCII"

datos_twitter['DISTRITOASCII'] = datos_twitter['Distrito'].apply(unidecode.unidecode)
datos_twitter['DISTRITOASCII'] = datos_twitter['DISTRITOASCII'].astype('str').str.lower()
from nltk.corpus import stopwords
stop = stopwords.words('spanish')
datos_twitter['DISTRITOASCII'] = datos_twitter['DISTRITOASCII'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
datos_twitter[['Distrito','DISTRITOASCII']]

# Producto cartesiano por distrito. 
# Como tengo la informacion por distrito y cruzo a nivel de barrio añado un nuevo registro por barrio.
datos_twitter = pd.merge(datos_twitter, datos_distrito_barrio, how='left', on='DISTRITOASCII')


"""# Evolución Precios"""

with hdfs.open(EvolucionPreciosAlquiler) as f:
  datos_evolucion_precios = pd.read_csv(f, sep=';')

# Los numeros tienen "","" en lugar de "".""
datos_evolucion_precios = datos_evolucion_precios.apply(lambda Precio: Precio.astype('str').str.replace(',','.'))

# En Precio hay caracteres raros... 
datos_evolucion_precios['Precio'] = pd.to_numeric(datos_evolucion_precios['Precio'], errors='coerce')

# excepciones
datos_evolucion_precios = datos_evolucion_precios.replace('Fuencarral-El Pardo', 'Fuencarral - El Pardo')
datos_evolucion_precios = datos_evolucion_precios.replace('Moncloa-Aravaca', 'Moncloa - Aravaca')
datos_evolucion_precios = datos_evolucion_precios.replace('San Blas-Canillejas', 'San Blas - Canillejas')


# Voy a agregas, me quito las columnas que no busco detalle
datos_evolucion_precios = datos_evolucion_precios.drop(['Anyo','Trimestre'],axis=1)
datos_evolucion_precios = datos_evolucion_precios.groupby(['Barrio']).agg({'Precio':'mean'}).reset_index()

# cruzo el distrito y saco su codigo.  "Barrio" = "DISTRITOASCII"

datos_evolucion_precios['DISTRITOASCII'] = datos_evolucion_precios['Barrio'].apply(unidecode.unidecode)
datos_evolucion_precios['DISTRITOASCII'] = datos_evolucion_precios['DISTRITOASCII'].astype('str').str.lower()
from nltk.corpus import stopwords
stop = stopwords.words('spanish')
datos_evolucion_precios['DISTRITOASCII'] = datos_evolucion_precios['DISTRITOASCII'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
datos_evolucion_precios[['Barrio','DISTRITOASCII']].head()

datos_evolucion_precios['DISTRITOASCII'].unique()

# Producto cartesiano por distrito. 
# Como tengo la informacion por distrito y cruzo a nivel de barrio añado un nuevo registro por barrio.
datos_evolucion_precios = pd.merge(datos_evolucion_precios, datos_distrito_barrio, how='left', on='DISTRITOASCII')

datos_evolucion_precios[datos_evolucion_precios['BARRIO'].isnull()]


"""# Transporte - Agregado"""
with hdfs.open(transporte_agregado) as f:
  datos_transporte = pd.read_csv(f)

datos_transporte.loc[:,'Distrito'] = datos_transporte.Barrio.astype('int64')
datos_transporte.loc[:,'Barrio'] = datos_transporte.Barrio.astype('int64')
datos_transporte = datos_transporte.drop(['Unnamed: 0'], axis = 1)
datos_transporte=datos_transporte.rename(columns = {'Distrito':'CODDIST'})
datos_transporte=datos_transporte.rename(columns = {'Barrio':'CODBAR'})


"""# Intervenciones Policia Municipal"""

# CUIDADO HE HECHO UNA NUEVA VERSION A MANO DEL FICHERO. Forzando el utf8, los EOF tipo unix y los odiosos puntos del millar. 

with hdfs.open(IntervencionesPMun_unix_utf8) as f:
  datos_intervenciones =  pd.read_csv(f, sep=';')

# excepciones
datos_intervenciones = datos_intervenciones.replace('Fuencarral-El Pardo', 'Fuencarral - El Pardo')
datos_intervenciones = datos_intervenciones.replace('Moncloa-Aravaca', 'Moncloa - Aravaca')
datos_intervenciones = datos_intervenciones.replace('San Blas-Canillejas', 'San Blas - Canillejas')

# columnas mal nombradas
datos_intervenciones=datos_intervenciones.rename(columns = {'Barrio':'distrito'})

# columnas que no agrego
datos_intervenciones = datos_intervenciones.drop(['Anyo'],axis=1)

datos_intervenciones = datos_intervenciones.groupby(['distrito']).agg('mean').reset_index()

# cruzo el distrito y saco su codigo.  "distrito" = "DISTRITOASCII"

datos_intervenciones['DISTRITOASCII'] = datos_intervenciones['distrito'].apply(unidecode.unidecode)
datos_intervenciones['DISTRITOASCII'] = datos_intervenciones['DISTRITOASCII'].astype('str').str.lower()
from nltk.corpus import stopwords
stop = stopwords.words('spanish')
datos_intervenciones['DISTRITOASCII'] = datos_intervenciones['DISTRITOASCII'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


datos_intervenciones = pd.merge(datos_intervenciones, datos_distrito_barrio, how='inner', on='DISTRITOASCII')

"""# Calidad del Aire"""

with hdfs.open(barrios_distritos_NO2) as f:
  datos_aire_NO2 =  pd.read_csv(f, sep=',')
datos_aire_NO2.columns =['CODBAR','BARRIO','CODDIST','Distrito', 'no2-media','no2-perc-5','no2-mediana','no2-perc-95']
datos_aire_NO2 = datos_aire_NO2.drop(['no2-perc-5','no2-mediana','no2-perc-95'], axis=1)


with hdfs.open(barrios_distritos_m25) as f:
  datos_aire_m25 = pd.read_csv(f, sep=',')
datos_aire_m25.columns =['CODBAR','BARRIO','CODDIST','Distrito', 'm25-media','m25-perc-5','m25-mediana','m25-perc-95']
datos_aire_m25 = datos_aire_m25.drop(['m25-perc-5','m25-mediana','m25-perc-95'], axis=1)


#'CODBAR','NAMBAR','CODDST','NAMDST'
datos_aire_m25['BARRIO'].unique()

"""#Colegios"""


# Funcion para saber a qué barrio de madrid pertenece una geoposicion
import json
from shapely.geometry import shape, Point

def get_barrio_distrito(lat, lon):
  with hdfs.open(barrios_edge_file) as f:
    js = json.load(f)
  point = Point(lon, lat)
  for feature in js['features']:
    polygon = shape(feature['geometry'])
    if polygon.contains(point):
        # print ('{} ubicado en el barrio {} del distrito {}'.format(point, feature['properties']['NOMBRE'], feature['properties']['NOMDIS']))
        return feature['properties']['CODBAR'],feature['properties']['NOMBRE'],feature['properties']['CODDISTRIT'],feature['properties']['NOMDIS']
        

with hdfs.open(colegios) as f:
  datos_colegios = pd.read_csv(f, sep=';')

# Calculo del barrio en dos fases
datos_colegios['BARCAL']=(datos_colegios.apply(lambda datos_colegios: get_barrio_distrito(datos_colegios['LATITUD'],datos_colegios['LONGITUD']),axis=1))

new_col_list = ['CODBAR','NOMBAR','CODDISTR','NOMDISTR']
for n, col in enumerate (new_col_list):
  datos_colegios[col]= datos_colegios['BARCAL'].apply(lambda BARCAR: BARCAR[n])

datos_colegios = datos_colegios.drop(['PK','NOMBRE','TRANSPORTE','DESCRIPCION','NOMBRE-VIA','CLASE-VIAL','TIPO-NUM',
 'COORDENADA-X',
 'COORDENADA-Y',
 'NUM','CODIGO-POSTAL','BARRIO','DISTRITO',
 'LATITUD',
 'LONGITUD',
 'BARCAL',],axis=1)

# En Precio hay caracteres raros... 
datos_colegios['CODBAR'] = pd.to_numeric(datos_colegios['CODBAR'], errors='coerce')
# agrego por barrio y distrito
datos_colegios = datos_colegios.groupby(['NOMBAR','CODBAR','NOMDISTR','CODDISTR']).size().reset_index(name='counts')
datos_colegios.columns=['NOMBAR','CODBAR','NOMDISTR','CODDISTR','colegios']

"""# Consolidación"""

# Datasets a consolidar:
# - datos_calidad_vida
# - datos_airbnb
# - datos_twitter
# - datos_evolucion_precios
# - datos_intervenciones
# - datos_aire_NO2
# - datos_aire_m25
# - datos_colegios




datos_calidad_vida['CODBAR'].nunique()

datos_aire_NO2['CODBAR'].nunique()

# calidad de vida con airbnb
datos_consolidados = pd.merge(datos_calidad_vida, datos_airbnb, how='inner', on='CODBAR')

# con twitter

datos_consolidados = pd.merge(datos_consolidados, datos_twitter, how='left', on='CODBAR')

# con la evolucion de precios

datos_consolidados = pd.merge(datos_consolidados, datos_evolucion_precios, how='inner', on='CODBAR')

# con datos_aire_NO2

datos_consolidados = pd.merge(datos_consolidados, datos_aire_NO2, how='inner', on='CODBAR')

# con datos_aire_m25

datos_consolidados = pd.merge(datos_consolidados, datos_aire_m25, how='inner', on='CODBAR')

# datos_colegios
datos_consolidados = pd.merge(datos_consolidados, datos_colegios, how='inner', on='CODBAR')

# datos_intervenciones
datos_consolidados = pd.merge(datos_consolidados, datos_intervenciones , how='left', on='CODBAR')

datos_consolidados = datos_consolidados.drop(['BARRIOASCII_x', 'CODDIST_x',
       'DISTRITO_x', 'CODBAR', 'BARRIO_x', 'DISTRITOASCII_x',
       'DISTRITO_y', 'CODDIST_y', 'BARRIO_y', 'DISTRITOASCII_y',
       'CODDIST_x', 'DISTRITO_x', 'BARRIO_x', 'BARRIOASCII_y', 'Barrio',
       'CODDIST_y', 'DISTRITO_y', 'BARRIO_y',
       'BARRIOASCII', 'BARRIO_x', 'CODDIST_x', 'Distrito_y','BARRIO_y',
       'CODDIST_y'],axis=1)


# quito columnas con NAN (las que solo tienen distrito)
datos_consolidados=datos_consolidados.dropna(axis=1)

datos_consolidados = datos_consolidados.drop_duplicates()

with hdfs.open(datos_consolidados_fin, 'wb') as f:
  datos_consolidados.to_csv(f)


