###  Python script for the Code Editor in QGIS  ###
###################################################

## Title: Create_point_objects.py
## Created: 4 August 2022
## Author: Rachid Oucheikh and Lars Harrie
## Description: Creates a layer containing all the points that are shown in the background map.

import csv
import codecs
import processing


#Open the output file where to save the results and write its header
output_file = open(path+'Readability_samples.csv', 'w', newline='')
writer = csv.writer(output_file)
writer.writerow(['Landmark Label'] + ['Manual: Label ID '] + ['Manual: Number of points '] + ['Manual: Number of lines'] + ["Manual: Overlap line length"] + ['CycleGAN: Label ID'] + ['CycleGAN: Number of points'] +
    ['CycleGAN: Number of lines'] + ["CycleGAN: Overlap line length"] 
      + ['Pix2pix: Label ID'] + ['Pix2pix: Number of points'] + ['Pix2pix: Number of lines'] + ["Pix2pix: Overlap line length"] + 
     ['QGIS: Label ID'] + ['QGIS: Number of points'] + ['QGIS: Number of lines'] + ["QGIS: Overlap line length"])

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
        if (labels_all[feat].geometry().area() > 150):
            labels.append(labels_all[feat])
    print('Number of labels obtained in the evaluation area:', len(labels))

    return labels_all, labels

def Load_ML_Labels(Labels_layer):
    # Get a list of all the landmark labels created using CycleGAN model
    labels_all = []
    for layer in Labels_layer:
        labels_all.append( [f for f in layer.getFeatures() ])
    labels_all = [ i for sub in labels_all for i in sub]
    #all the labels exist in the evaluation area
    print('Number of labels obtained in the evaluation area :', len(labels_all))

    return labels_all


#update the radability factor dictionary using the regional readability factor dictionary (regional = specific for an evaluation area) 
def update_dict(dict2, dict1):
    for key in dict1.keys():
        if key not in dict2.keys():
            dict2[key] = dict1[key]
        else:
            dict2[key] = min(dict1[key] , dict1[key])
    return dict2


def merge_results(point_dict, lines_dict):
    readability_dict = {}
    point_keys = point_dict.keys()
    lines_keys = lines_dict.keys()
    keys = list(set(list(point_keys) + list(lines_keys)))
    for key in keys:
        if (key in point_keys)  and (key in lines_keys):
            readability_dict[key] = [point_dict[key][0],point_dict[key][1], lines_dict[key][1], lines_dict[key][2]]
        elif key in point_keys:
            readability_dict[key] = [point_dict[key][0],point_dict[key][1], "NAN", "NAN"]
        elif key in lines_keys:
            readability_dict[key] = [lines_dict[key][0], "NAN", lines_dict[key][1], lines_dict[key][2]]

    return readability_dict


#Load all lines existing in the evaluation area 
def load_lines_prov(root, prov, evaluation_area):
    #Load the "Line objects" layer. There are two groups of line layers 
    mygroup = root.findGroup("Line objects")
    layers = mygroup.findLayers()
    Line_layer1 = [x.layer() for x in layers ]

    #The second groupe of feature was obtained from polygon conversion. 
    mygroup = root.findGroup("Line_features")
    layers = mygroup.findLayers()
    Line_layer2 = [x.layer() for x in layers ]
    print("Total number of layers is :", len(Line_layer2))

    layers = Line_layer1 + Line_layer2

    Lines_temp = []
    for l in layers:
        Lines_temp.append( [f for f in l.getFeatures() ])

    Lines_all = [i for sub in Lines_temp for i in sub]
    
    print("Total number of multilines is :", len(Lines_all))

    selected_features =[]
    for feat in Lines_all:
        if feat.geometry().intersects(evaluation_area):
            selected_features.append(feat)

    print("Total number of multilines in the evaluation area:", len(selected_features))
    
    count = 0
    list_lines = []
    for feat_i in selected_features:
        #feat_i = Lines_all[feat]
        count = count + 1
        geom = feat_i.geometry()
        mulitlines = geom.asMultiPolyline()
        #Extract all the lines from the layer
        if len(mulitlines) > 1:
            for j in range(0,len(mulitlines)-1):
                for k in range(0,len(mulitlines[j])-1):
                    line = QgsFeature()
                    line.setGeometry(QgsGeometry.fromPolylineXY([mulitlines[j][k],mulitlines[j][k+1]]))
                    line.setAttributes([count])
                    prov.addFeatures([line])
                    list_lines.append(line)
                    count += 1
        else:
            for i in range(0,len(mulitlines[0])-1):
                line = QgsFeature()
                line.setGeometry(QgsGeometry.fromPolylineXY([mulitlines[0][i],mulitlines[0][i+1]]))
                line.setAttributes([count])
                prov.addFeatures([line])
                list_lines.append(line)
                count += 1

    print("Total number of lines in the evaluation area:", len(list_lines))

    return list_lines, prov

def load_lines():
    #Load all the line segments obtained from conversion.
    Line_layer = QgsProject.instance().mapLayersByName('layer_linesA')


    Line_features = []
    for f in Line_layer[0].getFeatures():
        Line_features.append(f)

    return Line_features

#Load the layer containing the points and extract points 
def load_points(evaluation_area):
    #Load the point layer. 
    points_layer = QgsProject.instance().mapLayersByName('layer_points')

    point_features = []
    for f in points_layer[0].getFeatures():
        point_features.append(f)

    return point_features



def save_layer(root, evaluation_area):
    epsg = Line_layer[0].crs().postgisSrid()
    uri = "LineString?crs=epsg:" + str(epsg) + "&field=id:integer""&index=yes"

    line_layer = QgsVectorLayer(uri,
                               'layer_linesA',
                               'memory')

    prov = line_layer.dataProvider()
    prov.addAttributes([QgsField("objectID", QVariant.Int)])
    line_layer.updateFields()
    _, prov = load_lines(root, prov, evaluation_area)
    line_layer.updateExtents()
    QgsProject.instance().addMapLayer(line_layer)

    print("No. fields:", len(prov.fields()))
    print("No. features:", prov.featureCount())
    e = line_layer.extent()
    print("Extent:", e.xMinimum(), e.yMinimum(), e.xMaximum(), e.yMaximum())


def get_association_dict(spatialIndexStructure_landmark, labels, ID='ID'):
    # Create a dictionary that contain the linkage between landmark buildings and manual labels
    Associations_dict ={}

    # Go through all labels
    for feature_j in labels:
        feature_j_name = feature_j[ID]
        area = 0
        # Get the landmark features that overlap the label from the spatial index structure
        labelBB = feature_j.geometry().boundingBox()
        selected_features = spatialIndexStructure_landmark.intersects(labelBB)

        # Find the landmark feature that has the largest overlap with the label
        if len(selected_features) > 0:
            for feature_i_name in selected_features:
                old_area = area
                feature_i = features_landmark_all[feature_i_name - 1]  # Be aware of that the index refer to the list of all features
                intersection = feature_i.geometry().intersection(feature_j.geometry())
                area = intersection.area()
                if (area >= old_area):
                    Associations_dict[feature_j_name] = feature_i_name

        # If no landmark feature overlaps the label then the closest landmark is selected
        else:
            labelCentroid = feature_j.geometry().centroid()
            closest_label_feature = spatialIndexStructure_landmark.nearestNeighbor(labelCentroid, 1, 10000000)
            Associations_dict[feature_j_name] = int(closest_label_feature[0])

    return Associations_dict



#********************************************** Compute the radability factor using the intersection with line features **********************************************

def readability_lines(lines, spatialIndexStructure_lines, labels, dict_association, ID='ID'):  #labels can be labels obtained manually, using pix2pix, CycleGAN or optimization
    #Count the number of overlapped lines and their length    
    Readability_lines_factor = {}

    j=0
    for j in range(0, len(labels)-1):
        try:
            feature_j = labels[j]
            j += 1
            Label_ID = feature_j[ID]
            Landmark_ID = dict_association[Label_ID]

            # Get the lines that overlap the label from the spatial index structure
            labelBB = feature_j.geometry().boundingBox()
            selected_features = spatialIndexStructure_lines.intersects(labelBB)

            #For each label, we save the number of intersected points
            count_lines = len(selected_features)
            if count_lines != 0:
                length = 0.0
                #iterate over all the lines existing in the map to check their intersection with every signle label 
                for feat in selected_features:
                    #try:
                    feature_i = lines[feat]
                    intersection = feature_i.geometry().intersection(feature_j.geometry())
                    length = length + abs(intersection.length())

                Readability_lines_factor[Landmark_ID] = [Label_ID, count_lines, length]
                print("The number of lines and the corresponding length for landmark ", Landmark_ID , " are : ", Label_ID, count_lines, length)

            else:
                Readability_lines_factor[Landmark_ID] = [Label_ID, 0, 0.0]


        except Exception as e:                    
            print(e)

    return Readability_lines_factor



#********************************************** Compute the readability factor using the intersection with points  **********************************************
    
def readability_points(spatialIndexStructure_points, labels, dict_association, ID='ID'):
    Readability_points_factor = {}

    for feature_j in labels:
        Label_ID = feature_j[ID]
        try:
            Landmark_ID = dict_association[Label_ID]
            
            # Get the points that overlap the label from the spatial index structure
            labelBB = feature_j.geometry().boundingBox()
            selected_features = spatialIndexStructure_points.intersects(labelBB)

            #For each label, we save the number of intersected points
            count_points = len(selected_features)
            if count_points != 0:
                Readability_points_factor[Landmark_ID] = [Label_ID, count_points]

            if count_points == 0:
                Readability_points_factor[Landmark_ID] = [Label_ID, 0]
        except Exception as e:
            print(e)
            
    return Readability_points_factor


root = QgsProject.instance().layerTreeRoot()

mygroup = root.findGroup("Line objects")
layers = mygroup.findLayers()
Line_layer = [x.layer() for x in layers ]



# Define the evaluation area (defined as a polygon in the QGIS environment)
layer_evaluation_areas = QgsProject.instance().mapLayersByName('Test_Area')
for feature in layer_evaluation_areas[0].getFeatures():
    if feature['ID'] == 200:
        feature_evaluation  = feature

evaluation_area = feature_evaluation.geometry()
# An extended evaluation area is used for the landmark buildings
evaluationAreaPolygon_extended = evaluation_area.buffer(400,1)

# Get a list of all the landmark features (building objects)
layer_landmark = QgsProject.instance().mapLayersByName('Landmark')
features_landmark_all = []
for l in layer_landmark:
    features_landmark_all.append( [f for f in l.getFeatures() ])
features_landmark_all = [ i for sub in features_landmark_all for i in sub]
print('Number of landmark buildings in total:', len(features_landmark_all))
# Create a list with only those landmark features that are within the study area
features_landmark_study_area = []
for feature_j in features_landmark_all:
    if feature_j.geometry().intersects(evaluation_area):
        features_landmark_study_area.append(feature_j)
# Create a list with only those landmark features that are within an extended study area
features_landmark = []
for feature_j in features_landmark_all:
    if feature_j.geometry().intersects(evaluationAreaPolygon_extended):
        features_landmark.append(feature_j)
print('Number of landmark buildings in (extended) evaluation area:', len(features_landmark))

# Add the landmark to a spatial index data structure (R-tree)
spatialIndexStructure_landmark = QgsSpatialIndex()
spatialIndexStructure_landmark.addFeatures(features_landmark)

#Use the following line only if you need to save the line features
#save_layer(root, evaluation_area)


#********************************************** Load the lines and points existing in the evaluation area ****************************************
print("Loding line features...")
Line_features = load_lines()
print('Number of lines existing in the evaluation area:', len(Line_features))
spatialIndexStructure_lines = QgsSpatialIndex()
spatialIndexStructure_lines.addFeatures(Line_features)


print("Loding point features...")
Point_features = load_points(evaluation_area)
print('Number of points existing in the evaluation area:', len(Point_features))
spatialIndexStructure_points = QgsSpatialIndex()
spatialIndexStructure_points.addFeatures(Point_features)

Readability_points_factor, Readability_points_factor_manual, Readability_points_factor_CycleGan, Readability_points_factor_pix2pix, Readability_points_factor_QGIS = {}, {}, {}, {}, {}
Readability_lines_factor, Readability_lines_factor_manual, Readability_lines_factor_CycleGan, Readability_lines_factor_pix2pix, Readability_lines_factor_QGIS, Readability_factor_QGIS = {}, {}, {}, {}, {}, {}




#********************************************** Compute the readability factor for the manual labels **********************************************
# Create a list with only those manual labels that are within the study area
#Load the landmark layer that contains the landmark labels
layer_manual_labels = QgsProject.instance().mapLayersByName('LMF_Landmark_Building_T')

#Extract all the labels created manually and store them in a single list
labels_manually_all = []
for layer in layer_manual_labels:
    labels_manually_all.append( [f for f in layer.getFeatures() ])
labels_manually_all = [ i for sub in labels_manually_all for i in sub]
print('Number of manual labels in total:', len(labels_manually_all))

labels_manually = []
for feature_j in labels_manually_all:
    if feature_j.geometry().intersects(evaluation_area):
        labels_manually.append(feature_j)
print('Number of manual labels in evaluation area:', len(labels_manually))

Manual_association = get_association_dict(spatialIndexStructure_landmark, labels_manually, ID='OBJECTID')
print("Association dict", len(Manual_association))
Readability_points_factor = readability_points(spatialIndexStructure_points, labels_manually, Manual_association, ID='OBJECTID')
Readability_points_factor_manual = update_dict(Readability_points_factor_manual, Readability_points_factor)
print("Length of dictionary for points readability of manual labels is :", len(Readability_points_factor_manual)) 

Readability_lines_factor = readability_lines(Line_features, spatialIndexStructure_lines, labels_manually, Manual_association, ID='OBJECTID' )
Readability_lines_factor_manual = update_dict(Readability_lines_factor_manual, Readability_lines_factor)
print("Length of dictionary for lines readability of manual labels is :", len(Readability_lines_factor_manual)) 

Readability_factor_manual = merge_results(Readability_points_factor_manual, Readability_lines_factor_manual)







#********************************************** Compute the readability factor for the CycleGAN labels **********************************************
# Create a list with only those manual labels that are within the study area
# Associations of CycleGAN created labels for the first test set
layer_cycleGAN_labels = QgsProject.instance().mapLayersByName('CycleGAN_labels_1_Final')  # Labels obtained by CycleGAN for test set 1
#labels = Load_ML_Labels(layer_cycleGAN_labels)
labels_all, labels = extract_labels(evaluation_area, layer_cycleGAN_labels)
CycleGAN_association_1 = get_association_dict(spatialIndexStructure_landmark, labels)

Readability_points_factor = readability_points(spatialIndexStructure_points,  labels, CycleGAN_association_1)
Readability_points_factor_CycleGan = update_dict(Readability_points_factor_CycleGan, Readability_points_factor)
print("length of dictionary for points readability of CycleGAN labels is :", len(Readability_points_factor_CycleGan))

Readability_lines_factor = readability_lines(Line_features, spatialIndexStructure_lines, labels, CycleGAN_association_1, ID='ID')
Readability_lines_factor_CycleGan = update_dict(Readability_lines_factor_CycleGan, Readability_lines_factor)
print("length of dictionary for lines readability of CycleGAN labels is :", len(Readability_lines_factor_CycleGan))

# Associations of CycleGAN created labels for the second test set
layer_cycleGAN_labels = QgsProject.instance().mapLayersByName('CycleGAN_labels_2_Final')  # Labels obtained by CycleGAN for test set 1
#labels = Load_ML_Labels(layer_cycleGAN_labels)
labels_all, labels = extract_labels(evaluation_area, layer_cycleGAN_labels)
CycleGAN_association_2 = get_association_dict(spatialIndexStructure_landmark, labels)

Readability_points_factor = readability_points(spatialIndexStructure_points, labels, CycleGAN_association_2)
Readability_points_factor_CycleGan = update_dict(Readability_points_factor_CycleGan, Readability_points_factor)
print("length of dictionary for points readability of CycleGAN labels is :", len(Readability_points_factor_CycleGan))

Readability_lines_factor = readability_lines(Line_features, spatialIndexStructure_lines, labels, CycleGAN_association_2, ID='ID')
Readability_lines_factor_CycleGan = update_dict(Readability_lines_factor_CycleGan, Readability_lines_factor)
print("length of dictionary for lines readability of CycleGAN labels is :", len(Readability_lines_factor_CycleGan))

Readability_factor_CycleGan = merge_results(Readability_points_factor_CycleGan, Readability_lines_factor_CycleGan)
print("length of dictionary for readability of CycleGAN labels is :", len(Readability_factor_CycleGan))


#To save only the results for CycleGAN, use the following commented code
# count = 0
# for key in Readability_factor_manual.keys():
#     count += 1
#     if (key in Readability_factor_CycleGan.keys()):
#         writer.writerow([str(key)] + [str(Readability_factor_manual[key][0])] + [str(Readability_factor_manual[key][1])] + [str(Readability_factor_manual[key][2])] + [str(Readability_factor_manual[key][3])] 
#                 + [str(Readability_factor_CycleGan[key][0])] + [str(Readability_factor_CycleGan[key][1])] + [str(Readability_factor_CycleGan[key][2])] + [str(Readability_factor_CycleGan[key][3])] )

# output_file.close()




#********************************************** Compute the readability factor for the Pix2pix labels **********************************************
# Create a list with only those manual labels that are within the study area
# Associations of Pix2pix created labels for the first test set
layer_Pix2pix_labels = QgsProject.instance().mapLayersByName('Pix2pix_labels_1_Final')  # Labels obtained by Pix2pix for test set 1
labels_all, labels = extract_labels(evaluation_area, layer_Pix2pix_labels)
Pix2pix_association_1 = get_association_dict(spatialIndexStructure_landmark, labels, ID='id')

Readability_points_factor = readability_points(spatialIndexStructure_points, labels, Pix2pix_association_1, ID='id')
Readability_points_factor_pix2pix = update_dict(Readability_points_factor_pix2pix, Readability_points_factor)
print("length of dictionary for points readability of Pix2pix labels is :", len(Readability_points_factor_pix2pix))

Readability_lines_factor = readability_lines(Line_features, spatialIndexStructure_lines, labels, Pix2pix_association_1, ID='id')
Readability_lines_factor_pix2pix = update_dict(Readability_lines_factor_pix2pix, Readability_lines_factor)
print("length of updated dictionary for lines readability of Pix2pix labels is :", len(Readability_lines_factor_pix2pix))

# Associations of Pix2pix created labels for the second test set
layer_Pix2pix_labels = QgsProject.instance().mapLayersByName('Pix2pix_labels2_Final')  # Labels obtained by Pix2pix for test set 2
labels_all, labels = extract_labels(evaluation_area, layer_Pix2pix_labels)
Pix2pix_association_2 = get_association_dict(spatialIndexStructure_landmark, labels, ID='id')

Readability_points_factor = readability_points(spatialIndexStructure_points, labels, Pix2pix_association_2, ID='id')
Readability_points_factor_pix2pix = update_dict(Readability_points_factor_pix2pix, Readability_points_factor)
print("length of dictionary for points readability of Pix2pix labels is :", len(Readability_points_factor_pix2pix))

Readability_lines_factor = readability_lines(Line_features, spatialIndexStructure_lines, labels, Pix2pix_association_2, ID='id')
Readability_lines_factor_pix2pix = update_dict(Readability_lines_factor_pix2pix, Readability_lines_factor)
print("length of uodated dictionary for lines readability of Pix2pix labels is :", len(Readability_lines_factor_pix2pix))


Readability_factor_pix2pix = merge_results(Readability_points_factor_pix2pix, Readability_lines_factor_pix2pix)
print("length of updated dictionary for readability of Pix2pix labels is :", len(Readability_factor_pix2pix))



#********************************************** Compute the readability factor for the QGIS labels **********************************************
# Create a list with only those QGIS labels that are within the study area
#Load the landmark layer that contains the landmark labels

layer_QGIS = QgsProject.instance().mapLayersByName('layer_QGIS_all')
labels_all, labels = extract_labels(evaluation_area,layer_QGIS)
print('Total number of labels obtained using QGIS-PAL for the test area:', len(labels_all))
print('Number of labels obtained using QGIS-PAL for the test area:', len(labels))
Associations_QGIS_dict = get_association_dict(spatialIndexStructure_landmark, labels, ID='id')
print('Number of key-value pairs in the dictionary for QGIS-PAL labels:', len(Associations_QGIS_dict.keys()))


Readability_points_factor = readability_points(spatialIndexStructure_points, labels, Associations_QGIS_dict, ID='id')
Readability_points_factor_QGIS = update_dict(Readability_points_factor_QGIS, Readability_points_factor)
print("Length of dictionary for points readability of QGIS labels is :", len(Readability_points_factor_QGIS)) 

Readability_lines_factor = readability_lines(Line_features, spatialIndexStructure_lines, labels, Associations_QGIS_dict, ID='id' )
Readability_lines_factor_QGIS = update_dict(Readability_lines_factor_QGIS, Readability_lines_factor)
print("Length of dictionary for lines readability of QGIS labels is :", len(Readability_lines_factor_QGIS)) 

Readability_factor_QGIS = merge_results(Readability_points_factor_QGIS, Readability_lines_factor_QGIS)



#**************************************************************** Save the result  **********************************************************************



#Save results 
for key in Readability_factor_manual.keys():
    if key in Readability_factor_CycleGan.keys():
        cycle_label, cycle_points, cycle_lines, cycle_length  = str(Readability_factor_CycleGan[key][0]) , str(Readability_factor_CycleGan[key][1]), str(Readability_factor_CycleGan[key][2]) , str(Readability_factor_CycleGan[key][3])
    else:
        cycle_label, cycle_points, cycle_lines, cycle_length = "NAN", "NAN", "NAN", "NAN"
    if key in Readability_factor_pix2pix.keys():
        pix_label, pix_points, pix_lines, pix_length  = str(Readability_factor_pix2pix[key][0]) , str(Readability_factor_pix2pix[key][1]), str(Readability_factor_pix2pix[key][2]) , str(Readability_factor_pix2pix[key][3])
    else:
        pix1, pix2 = "NAN", "NAN", "NAN", "NAN"
    if key in Readability_factor_QGIS.keys():
        qgis_label, qgis_points, qgis_lines, qgis_length = str(Readability_factor_QGIS[key][0]) , str(Readability_factor_QGIS[key][1]), str(Readability_factor_QGIS[key][2]) , str(Readability_factor_QGIS[key][3])
    else:
        qgis_label, qgis_points, qgis_lines, qgis_length = "NAN", "NAN", "NAN", "NAN"

    writer.writerow([str(key)] + [str(Readability_factor_manual[key][0])] + [str(Readability_lines_factor_manual[key][1])] + [str(Readability_lines_factor_manual[key][2])] + [str(Readability_lines_factor_manual[key][3])] 
        +  [cycle_label]  + [cycle_points]  + [cycle_lines]  + [cycle_length]  + [pix_label]  + [pix_points]  + [pix_lines]  + [pix_length]  + [qgis_label] + [qgis_points] + [qgis_lines] + [qgis_length] )

output_file.close()

#To save only the results for the common labels existing on all the methods, use the following code
# count = 0
# for key in Readability_factor_manual.keys():
#     count += 1
#     if (key in Readability_factor_CycleGan.keys()) and (key in Readability_factor_pix2pix.keys()) and (key in Readability_factor_QGIS.keys()):
#         writer.writerow([str(key)] + [str(Readability_factor_manual[key][0])] + [str(Readability_factor_manual[key][1])] + [str(Readability_factor_manual[key][2])] + [str(Readability_factor_manual[key][3])] 
#                 + [str(Readability_factor_CycleGan[key][0])] + [str(Readability_factor_CycleGan[key][1])] + [str(Readability_factor_CycleGan[key][2])] + [str(Readability_factor_CycleGan[key][3])] + 
#                 [str(Readability_factor_pix2pix[key][0])] + [str(Readability_factor_pix2pix[key][1])] + [str(Readability_factor_pix2pix[key][2])] + [str(Readability_factor_pix2pix[key][3])] + 
#              [str(Readability_factor_QGIS[key][0])] + [str(Readability_factor_QGIS[key][1])] + [str(Readability_factor_QGIS[key][2])] + [str(Readability_factor_QGIS[key][3])] )

# output_file.close()
