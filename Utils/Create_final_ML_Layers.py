###  Python script for the Code Editor in QGIS  ###
###################################################

## Title: Final_ML_Layers_Creation.py
## Created: 13 September 2022
## Description: Creates the final layers by merging the labels from different sets that have the best association factors. These layers are used for the legibility evaluation so to have a fair comparison.

import csv


#Function to extract features existing inside an evaluation area from given layers 
def extract_labels(evaluationArea, Labels_layer):
    # Get a list of all the landmark labels created using CycleGAN model
    labels_all = []
    for layer in Labels_layer:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]
    print('The total number of all labels is :', len(labels_all))
    # Create a list with only those landmark features that are within the study area
    spatialIndexStructure_temp = QgsSpatialIndex()
    spatialIndexStructure_temp.addFeatures(labels_all)
    # Get the lines that overlap the label from the spatial index structure
    labelBB = evaluationArea.boundingBox()
    selected_features = spatialIndexStructure_temp.intersects(labelBB) 
    labels = []
    for feat in selected_features:
        #if (labels_all[feat].geometry().area() > 150):
        labels.append(labels_all[feat-1])
    print('Number of labels obtained in the evaluation area:', len(labels))

    return labels_all, labels

path = "Path to the results folder"

root = QgsProject.instance().layerTreeRoot()
mygroup = root.findGroup("Line objects")
layers = mygroup.findLayers()
Line_layer = [x.layer() for x in layers ]

epsg = Line_layer[0].crs().postgisSrid()
uri = "polygon?crs=epsg:" + str(epsg) + "&field=id:integer""&index=yes"

# Define the evaluation area (defined as a polygon in the QGIS environment)
layer_evaluation_areas = QgsProject.instance().mapLayersByName('Test_Area')
for feature in layer_evaluation_areas[0].getFeatures():
    if feature['ID'] == 0:
        feature_evaluation  = feature

evaluation_area = feature_evaluation.geometry()

def create_Pix2pix_labels():
    labels_layer = QgsVectorLayer(uri,
                               'Final_Pix2pix_layer',
                               'memory')

    prov = labels_layer.dataProvider()
    prov.addAttributes([QgsField("objectID", QVariant.Int)])
    labels_layer.updateFields()

    file2= path + '/Labels_for_Legibility_Pix2Pix.csv'

    Associations_dict_Pix2pix = {}

    i=1
    with open(file2, "r") as f2:
        reader2 = csv.reader(f2,delimiter=',')
        for line in reader2:
            if len(line) != 0:
                Associations_dict_Pix2pix[i] = [int(a) for a in line]
                i += 1

    layer_Pix2pix_labels = QgsProject.instance().mapLayersByName('Pix2pix_labels_1_Final')  # Labels obtained by CycleGAN for test set 1
    #labels = Load_ML_Labels(layer_cycleGAN_labels)
    labels_all_Pix2pix, labels_Pix2pix = extract_labels(evaluation_area, layer_Pix2pix_labels)
    labels_Pix2pix = [label for label in labels_Pix2pix if label['ID'] in Associations_dict_Pix2pix[1]]

    layer_Pix2pix_labels = QgsProject.instance().mapLayersByName('Pix2pix_labels_2_Final')  # Labels obtained by CycleGAN for test set 2
    #labels = Load_ML_Labels(layer_cycleGAN_labels)
    labels_all_2, labels_2 = extract_labels(evaluation_area, layer_Pix2pix_labels)
    labels_2 = [label for label in labels_2 if label['ID'] in Associations_dict_Pix2pix[2]]
    labels_Pix2pix.extend(labels_2)
    labels_all_Pix2pix.extend(labels_all_2)

    print("The total number of labels for the landmarks after processing is : ",len(labels_all_Pix2pix))
    layer_Pix2pix_labels = QgsProject.instance().mapLayersByName('Pix2pix_Roads_processed')  
    labels_all_roads, labels_roads = extract_labels(evaluation_area, layer_Pix2pix_labels)
    print("The total number of labels for the roads after processing is : ", len(labels_all_roads))
    labels_Pix2pix.extend(labels_roads)
    labels_all_Pix2pix.extend(labels_all_roads)
    print("The total number of labels for the landmarks and roads after processing is :", len(labels_all_Pix2pix))


    count = 0
    for label in labels_Pix2pix:
        f = QgsFeature()
        f.setGeometry(label.geometry())
        f.setAttributes([count])
        prov.addFeatures([f])
        count += 1

    labels_layer.updateExtents()
    QgsProject.instance().addMapLayer(labels_layer)

    print('Number of labels identified in layers in the evaluation area', count)
    print("No. fields:", len(prov.fields()))
    print("No. features:", prov.featureCount())
    e = labels_layer.extent()
    print("Extent:", e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum())


def create_Pix2pix_labels():
    labels_layer = QgsVectorLayer(uri,
                               'Final_CycleGAN_layer',
                               'memory')

    prov = labels_layer.dataProvider()
    prov.addAttributes([QgsField("objectID", QVariant.Int)])
    labels_layer.updateFields()


    file2= path + '/Labels_for_Legibility_CycleGAN.csv'

    Associations_dict_GAN = {}

    i=1
    with open(file2, "r") as f2:
    	reader2 = csv.reader(f2,delimiter=',')
    	for line in reader2:
    		if len(line) != 0:
    			Associations_dict_GAN[i] = [int(a) for a in line]
    			i += 1

    duplicates = [l for l in Associations_dict_GAN[1] if l in Associations_dict_GAN[2]]

    layer_cycleGAN_labels = QgsProject.instance().mapLayersByName('CycleGAN_labels_1_Final')  # Labels obtained by CycleGAN for test set 1
    #labels = Load_ML_Labels(layer_cycleGAN_labels)
    labels_all_QGIS, labels_QGIS = extract_labels(evaluation_area, layer_cycleGAN_labels)
    labels_QGIS = [label for label in labels_QGIS if label['ID'] in Associations_dict_GAN[1]]

    layer_cycleGAN_labels = QgsProject.instance().mapLayersByName('CycleGAN_labels_2_Final')  # Labels obtained by CycleGAN for test set 2
    #labels = Load_ML_Labels(layer_cycleGAN_labels)
    labels_all_2, labels_2 = extract_labels(evaluation_area, layer_cycleGAN_labels)
    labels_2 = [label for label in labels_2 if label['ID'] in Associations_dict_GAN[2]]
    labels_QGIS.extend(labels_2)
    labels_all_QGIS.extend(labels_all_2)

    print( "The total number of labels for the landmarks after processing is : " ,len(labels_all_QGIS))
    layer_cycleGAN_labels = QgsProject.instance().mapLayersByName('CycleGAN_Roads_Final')  
    labels_all_roads, labels_roads = extract_labels(evaluation_area, layer_cycleGAN_labels)
    print(len(labels_all_roads))
    labels_QGIS.extend(labels_roads)
    labels_all_QGIS.extend(labels_all_roads)
    print("The total number of labels for the roads after processing is : ", len(labels_all_QGIS))
    print("The total number of labels for the landmarks and roads after processing is :", len(labels_QGIS), len(labels_all_QGIS))

    count = 0
    for label in labels_QGIS:
        f = QgsFeature()
        f.setGeometry(label.geometry())
        f.setAttributes([count])
        prov.addFeatures([f])
        count += 1         

    labels_layer.updateExtents()
    QgsProject.instance().addMapLayer(labels_layer)

    print('Number of labels identified in layers in the evaluation area', count)
    print("No. fields:", len(prov.fields()))
    print("No. features:", prov.featureCount())
    e = labels_layer.extent()
    print("Extent:", e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum())


