#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform standard from pamDP to camtraDP to load to GBIF and CSA event format

"""

import os
import pandas as pd
import datetime
import json


def from_media_to_media_gbif(media):
    """
    Conversion steps
    1. Drop audio columns that are not part of the GBIF media format
    2. Create a new column mediaComments that is a JSON object with the dropped columns
    
    Return the media_gbif dataframe with the new column and the dropped columns
    """
    columns_to_drop=['sampleRate' , 'bitDepth' , 'fileLength' , 'numChannels']
    media['mediaComments']= json.loads(media[columns_to_drop].T.to_json()).values()
    media_gbif = media.drop(columns=columns_to_drop)
    return media_gbif
    
def from_deployments_to_deployments_gbif(deployments):
    """
    Conversion steps

    1. Rename columns to match GBIF format:
       - recorderID       -> cameraID
       - recorderModel    -> cameraModel
       - recorderHeight   -> cameraHeight
       - recorderDepth    -> cameraDepth
       - recorderTilt     -> cameraTilt
       - recorderHeading  -> cameraHeading
    2. Add information from column recorderConfiguration to column deploymentTags
       as key:pair value, keeping the value that might be present in this column
       and splitting values with pipe character for multiple values
       key1:pair1 | key2:pair2, etc.
    """
    deployments_gbif = deployments.rename(
        columns={
            "recorderID": "cameraID",
            "recorderModel": "cameraModel",
            "recorderHeight": "cameraHeight",
            "recorderDepth": "cameraDepth",
            "recorderTilt": "cameraTilt",
            "recorderHeading": "cameraHeading",
        }
        )
    

    
    def _merge_tags(row):
        existing = row.get("deploymentTags")
        new_tags = row.get("recorderConfiguration")

        parts = []
        if isinstance(existing, str) and existing.strip():
            parts.append(existing.strip())
        if new_tags:
            parts.append(new_tags)

        return " | ".join(parts) if parts else existing

    deployments_gbif["deploymentTags"] = deployments_gbif.apply(_merge_tags, axis=1)

    # drop recorderConfiguration column
    deployments_gbif.drop(columns="recorderConfiguration", inplace=True)

    return deployments_gbif

def from_observations_to_observations_gbif(observations, media):
    """
    Conversion steps

    1. eventStart and eventEnd
       In camtrapDP these are absolute datetime values in ISO format. In pamDP
       they are given as seconds relative to the timestamp of the audio
       recording (media). To convert:
         - merge observations with media on mediaID to bring in the
           recording's timestamp column
         - add eventStart/eventEnd (seconds) to that timestamp as a timedelta
           to get the absolute datetime
         - format the result back to ISO 8601 to match camtrapDP
    2. Assign observationLevel as 'media'
    """
    dwc_observations = observations.merge(
        media[["mediaID", "timestamp"]],
        on="mediaID",
        how="left",
    )

    dwc_observations["timestamp"] = pd.to_datetime(dwc_observations["timestamp"])

    dwc_observations["eventStart"] = (
        dwc_observations["timestamp"]
        + pd.to_timedelta(dwc_observations["eventStart"], unit="s")
    ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    dwc_observations["eventEnd"] = (
        dwc_observations["timestamp"]
        + pd.to_timedelta(dwc_observations["eventEnd"], unit="s")
    ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    dwc_observations = dwc_observations.drop(columns=["timestamp"])

    dwc_observations = dwc_observations.assign(observationLevel="media")

    return dwc_observations

def from_deployments_to_CSA_eventos(deployments, media, fdm):
    # ---------------
    # ---------------
    # --CSA Columns--
    # ---------------
    # ---------------
    mandatory_columns_CSA_eventos = [
        "exist",
        "projectName",
        "MediaType",
        "RecordingEquipment",
        "SamplingRate",
        "Resolution",
        "TypeOfRecording",
        "MicrophoneTrademark",
        "IsCurrent1",
        "Country",
        "State",
        "County",
        "Habitat",
        "HabitatCharacteristics",
        "MinimumEleveation",
        "Latitude",
        "Longitude",
        "GeodeticDatum",
        "geolocationDevice",
        "StartDate",
        "CollectorFirstName1",
        "CollectorLastName1",
        "PreparedFirstName1",
        "PreparedLastName1",
    ]
    non_mandatory_columns_CSA_eventos = [
        "FieldNumber",
        "CatalogNumber",
        "CatalogerLastName",
        "CatalogerFirstName",
        "catalogedDate",
        "FolderLocation",
        "Duration(HH:MM:SS)",
        "FolderLocation",
        "PublishedRepository",
        "CommentsOfTheRecording",
        "Kingdom",
        "VerbatimLocality",
        "NationalParkName",
        "EndDate",
        "EventTime",
        "recorderHeight",
        "CollectingMethod",
        "eventRemarks",
        "PrepType1",
        "numberOfFiles",
        "Description1",
        "OtherCatalogNumber1",
    ]
    columnas_CSA_eventos = (
        mandatory_columns_CSA_eventos + non_mandatory_columns_CSA_eventos
    )

    # --------------------
    # --------------------
    # --Renaming Columns--
    # --------------------
    # --------------------

    deployments_rename_dictionary = {
        "latitude": "Latitude",
        "longitude": "Longitude",
        "recorderModel": "MicrophoneTrademark",
    }
    media_rename_dictionary = {"sampleRate": "SamplingRate", "bitDepth": "Resolution"}

    fdm_rename_dictionary = {
            "Ubicación en el medio de almacenamiento": "FolderLocation",
            "Indicador de evento": "FieldNumber",
            "Nombre de la carpeta proyecto (NOMBRE_NÚMEROIAVH)": "projectName",
            "Equipo de grabación": "RecordingEquipment",
            "Medio de almacenamiento temporal": "MediaType",
            "Comentario de sonido": "CommentsOfTheRecording",
            "País": "Country",
            "Departamento": "State",
            "Municipio": "County",
            "Localidad": "VerbatimLocality",
            "Área Natural Protegida": "NationalParkName",
            "Hábitat": "Habitat",
            "Características del hábitat": "HabitatCharacteristics",
            "Elevación": "MinimumEleveation",
            "Instrumento de geolocalización": "geolocationDevice",
            "Fecha inicial": "StartDate",
            "Fecha final": "EndDate",
            "Altura de la grabadora respecto al suelo": "recorderHeight",
            "Configuración de muestreo": "CollectingMethod",
            "Nombre del instalador": "CollectorFirstName1",
            "Apellido  del instalador": "CollectorLastName1",
            "Numero de archivos": "numberOfFiles",
        }

    fdm = fdm[
        list(fdm_rename_dictionary.keys()) + ["Hora inicial", "Hora final"]
    ].rename(columns=fdm_rename_dictionary)
    fdm["EventTime"] = (
        fdm["Hora inicial"].astype(str) + " | " + fdm["Hora final"].astype(str)
    )
    fdm = fdm.drop(columns=["Hora inicial", "Hora final"])

    media = media[
        list(media_rename_dictionary.keys()) + ["deploymentID", "fileLength"]
    ].rename(columns=media_rename_dictionary)
    deployments = deployments[
        list(deployments_rename_dictionary.keys()) + ["deploymentID"]
    ].rename(columns=deployments_rename_dictionary)
    media=media.groupby("deploymentID").agg(SamplingRate=('SamplingRate','first'),
                                    Resolution=('Resolution','first'),
                                    fileLength=('fileLength','sum')
                                    ).reset_index()
    media['fileLength']=media['fileLength'].astype('int64').apply(lambda x: f"""{x//(60*60):02d}:{(x%(60*60))//60:02d}:{(x%(60*60))%60:02d}""")
    media=media.rename(columns={'fileLength':"Duration(HH:MM:SS)"  })
    # -------------------
    # -------------------
    # --Join DataFrames--
    # -------------------
    # -------------------


    deployments = deployments.merge(
        media, on="deploymentID", how="left"
    )
    deployments = deployments.rename(columns={"deploymentID": "FieldNumber"})

    CSA_eventos = deployments.merge(fdm, on="FieldNumber", how="left")
    CSA_eventos['FolderLocation']=CSA_eventos[['FolderLocation','FieldNumber']].apply(lambda x: os.path.join(x['FolderLocation'],x['FieldNumber']),
                                                        axis=1
                                                    )
    columnas_fdm = [
        "Nombre de la carpeta proyecto (NOMBRE_AÑO)",
        "Indicador de evento",
        "Fecha inicial",
        "Fecha final",
        "País",
        "Departamento",
        "Municipio",
        "Localidad",
        "Latitud",
        "Longitud",
        "Numero de archivos",
        "Nombre del responsable",
        "Apellido  del responsable",
        "Equipo de grabación",
        "Definiciones!D17",
        "Elevación",
        "Calidad de grabación",
        "Nombre del proyecto",
        "Comentario de sonido",
        "Duración de cada grabación",
        "Configuración de muestreo",
        "Hábitat",
        "Área Natural Protegida",
        "Nombre del instalador",
        "Apellido  del instalador",
        "Publicado",
        "Estrato de Vegetación",
    ]
    # Broadcasted columns
    CSA_eventos["exist"] = "Yes"
    CSA_eventos["TypeOfRecording"] = "Monitoreo Acústico Pasivo"
    CSA_eventos["GeodeticDatum"] = "WGS84"
    CSA_eventos["Description1"] = "WAV"
    CSA_eventos["Kingdom"] = "Animalia"
    CSA_eventos["PrepType1"] = "Bloque de audios"
    CSA_eventos["IsCurrent1"] = "Yes"

    # Manually set by curator
    CSA_eventos["CatalogNumber"] = None
    CSA_eventos["OtherCatalogNumber1"] = None
    CSA_eventos["catalogedDate"] = None
    CSA_eventos["PublishedRepository"] = None
    CSA_eventos["CatalogerFirstName"] = None
    CSA_eventos["CatalogerLastName"] = None
    CSA_eventos["PreparedFirstName1"] = None
    CSA_eventos["PreparedLastName1"] = None
    CSA_eventos["eventRemarks"] = None
    if set(CSA_eventos.columns) != set(columnas_CSA_eventos):
        if set(CSA_eventos.columns).issubset(set(columnas_CSA_eventos)):
            raise ValueError(
                f"Missing columns for pamDP.Media format: \n list of missing columns {set(columnas_CSA_eventos) - set(CSA_eventos.columns)}"
            )
        elif set(columnas_CSA_eventos).issubset(CSA_eventos.columns):
            raise ValueError(
                f"Extra columns for pamDP.Media format: \n The following columns are not part of pamDP.Media format {set(CSA_eventos.columns) - set(columnas_CSA_eventos)}"
            )
        else:
            raise ValueError(f"""Column mismatch. There are extra and missing columns for pamDP.Media format. \n Expected columns: {columnas_CSA_eventos} 
                \n Missing columns: {set(columnas_CSA_eventos) - set(CSA_eventos.columns)} 
                \n Extra Columns{set(CSA_eventos.columns) - set(columnas_CSA_eventos)}""")
    return CSA_eventos