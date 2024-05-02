import cv2
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt
class TwoDctRockgeometry:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.max_area_contour = self.get_max_area_contour()
    def get_image(self):
        return Image.fromarray(self.image)
    ## Convexity/Convexidade
    def convexity(self):
        max_area = 0
        max_contour = None

        for cont in self.contours:
            print(len(self.contours))
            area = cv2.contourArea(cont)
            if area > max_area:
                max_area = area
                max_contour = cont

        if max_contour is not None:
            convex_area = cv2.contourArea(cv2.convexHull(max_contour))
            return max_area / convex_area if convex_area != 0 else 0
        else:
            return 0

    def get_max_area_contour(self):
        max_area = 0
        max_contour = None
        for cont in self.contours:
            # print(len(cont),"/",len(self.contours))
            print(len(cont))
            # print(len(cont))
            area = cv2.contourArea(cont)
            # if area > max_area:
            #     max_area = area
            #     max_contour = cont
            lencont=len(cont)
            if lencont > max_area:
                max_area = lencont
                max_contour = cont
        print("Max area contour:",len(max_contour))        
        return max_contour
    
    def draw_all_contours(self, image):
        drawn_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(self.contours):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            if i == len(self.contours) - 1:
                color = (0, 0, 255)  # Red color for the largest contour
            cv2.drawContours(drawn_image, [contour], -1, color, 2)
        return drawn_image
    ## Convexity/Convexidade
    def convexity(self):
        if self.max_area_contour is not None:
            area = cv2.contourArea(self.max_area_contour)
            convex_area = cv2.contourArea(cv2.convexHull(self.max_area_contour))
            return area / convex_area if convex_area != 0 else 0
        else:
            return 0

    ## Circularity/Circularidade
    def circularity(self):
        if self.max_area_contour is not None:
            # area = cv2.contourArea(self.max_area_contour)
            (x, y), radius = cv2.minEnclosingCircle(self.max_area_contour)

            perimeter = cv2.arcLength(self.max_area_contour, True)
            print("Perimeter:",perimeter)
            return (4 * np.pi*np.pi*radius**2) /( (2*np.pi*radius)**2) if perimeter != 0 else 0
    #Calculate the radius of the largest inscribed circle
        
        
        
        else:
            return 0

    # Intercept sphericity/Esfericidade interceptada
    def intercept_sphericity(self):
  # Calculate the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(self.max_area_contour)
        area = cv2.contourArea(self.max_area_contour)
        largest_contour = max(self.contours, key=cv2.contourArea)
        
        # area=cv2.minEnclosingCircle(largest_contour)
        print("Area:",area)
            # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(self.max_area_contour, True)
            
            # Calculate the intercept sphericity
        if perimeter != 0:
            return  ((y)/(radius+x) )        
    def radial_variance(self):
            if self.max_area_contour is not None:
                # Get the minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(self.max_area_contour)

                # Calculate distances from the center to each contour point
                distances = []
                for point in self.max_area_contour:
                    point_distance = np.sqrt((point[0][0] - x) ** 2 + (point[0][1] - y) ** 2)
                    distances.append(point_distance)

                # Calculate the variance of these distances
                distance_array = np.array(distances)
                variance = np.var(distance_array)

                # Normalize the variance by the radius to make the metric scale-invariant
                normalized_variance = variance / (radius ** 2) if radius != 0 else float('inf')

                return normalized_variance
            else:
                return float('inf') 
  
    ## Area Sphericity/Esfericidade por area
    def area_sphericity(self):
        if self.max_area_contour is not None:
            largest_contour = max(self.contours, key=cv2.contourArea)
            area = cv2.contourArea(self.max_area_contour)
            areacirc=cv2.minEnclosingCircle(largest_contour)
            print()
            return area / (np.pi * areacirc[1]**2) if areacirc[1] != 0 else 0

            circle = cv2.fit
        else:
            return 0

    ## Diameter sphericity/Diâmetro esfericidade
    def diameter_sphericity(self):
        if self.max_area_contour is not None:
            area = cv2.contourArea(self.max_area_contour)
            min_circle_radius = cv2.minEnclosingCircle(self.max_area_contour)[1]
            equivalent_diameter = 2 * np.sqrt(area / np.pi)
            return equivalent_diameter / (2 * min_circle_radius) if min_circle_radius != 0 else 0
        else:
            return 0

    # Circle ratio sphericity/Ligação da esfericidade do círculo
    def circle_ratio_sphericity(self):
        if self.max_area_contour is not None:
            area = cv2.contourArea(self.max_area_contour)
            min_circle_radius = cv2.minEnclosingCircle(self.max_area_contour)[1]
            max_inscribed_radius = min_circle_radius
            for i in range(int(min_circle_radius), 0, -1):
                if cv2.pointPolygonTest(self.max_area_contour, (int(min_circle_radius), int(min_circle_radius)), True) > 0:
                    max_inscribed_radius = i
                    break
            return max_inscribed_radius / min_circle_radius if min_circle_radius != 0 else 0
        else:
            return 0
   
    # def circle_ratio_sphericity(self):
    #         if self.max_area_contour is not None:
    #             (x, y), radius = cv2.minEnclosingCircle(self.max_area_contour)
    #             perimeter = cv2.arcLength(self.max_area_contour, True)
    #             return (4 * np.pi**2 * radius**2) / (perimeter**2) if perimeter != 0 else 0
    #         else:
    #             return 0
 

    ## Surface area sphericity/Esfericidade da superfície
    def surface_area_sphericity(self):
        if self.max_area_contour is not None:
                # Calculate the area of the contour
                area = cv2.contourArea(self.max_area_contour)

                ellipse = cv2.fitEllipse(self.max_area_contour)
                major_axis, minor_axis = ellipse[1]

                # Calculate the perimeter of the contour
                radiocircle_area = cv2.minEnclosingCircle(self.max_area_contour)[1]
                print("Minimun circle radio:",radiocircle_area)
                perimeter = cv2.arcLength(self.max_area_contour, True)
                radiocircle_area2=4*(area)/(np.pi*major_axis**2)
                #np.pi*radiocircle_area**2
                # Ensure the perimeter is not zero to avoid division by zero
                # if perimeter != 0:
                #     # Calculate the radius of the circle with the same perimeter as the contour
                #     radius = perimeter / (2 * np.pi)

                #     # Calculate the area of such a circle
                #     circle_area = np.pi * radius ** 2

                #     # Calculate sphericity based on the ratio of the contour area to the circle area
                #     sphericity = (area / circle_area)
                #sphericity =  (area)/(np.pi *radiocircle_area2**2)  if radiocircle_area1 != 0 else 0

                return radiocircle_area2
                # else:
                #     return 0
        else:
                return 0

    ## Aspect ratio/Proporção de aspeto
    def aspect_ratio(self):
        if self.max_area_contour is not None:
            if len(self.max_area_contour) >= 5:
                ellipse = cv2.fitEllipse(self.max_area_contour)
                major_axis, minor_axis = ellipse[1]
                print("Major axis:",max(minor_axis,major_axis))
                print("Minor axis:",min(minor_axis,major_axis))
                aspect_ratio = min(major_axis, minor_axis) / max(major_axis, minor_axis)
            else:
                aspect_ratio = 1
        else:
            aspect_ratio = 1

        return aspect_ratio
        
    
rock_geometry = TwoDctRockgeometry('/Users/victorvillegassalabarria/Documents/Mastria/FEM2024/TEXTODISSERTAÇÅO/ImagesForTestingShapeDescriptors/Buraco1.jpg')
# rock_geometry = TwoDctRockgeometry('/Users/victorvillegassalabarria/Documents/Mastria/FEM2024/TEXTODISSERTAÇÅO/ImagesForTestingShapeDescriptors/Buraco1.jpg')
# rock_geometry = TwoDctRockgeometry('/Users/victorvillegassalabarria/Documents/Mastria/FEM2024/TEXTODISSERTAÇÅO/ImagesForTestingShapeDescriptors/Buraco1.jpg')
# rock_geometry = TwoDctRockgeometry('/Users/victorvillegassalabarria/Documents/Mastria/FEM2024/TEXTODISSERTAÇÅO/ImagesForTestingShapeDescriptors/Buraco1.jpg')
namesgeoms=["circuloblanco.jpg","Buraco1.jpg","Buraco2.jpg","Fract1.jpg","Fract2.jpg"]
for i in range(5):
    rock_geometry = TwoDctRockgeometry('/Users/victorvillegassalabarria/Documents/Mastria/FEM2024/TEXTODISSERTAÇÅO/ImagesForTestingShapeDescriptors/'+namesgeoms[i])
        # print("Circularity:", rock_geometry.circularity())

    print("Results for image :",namesgeoms[i])
    print("Convexity:", rock_geometry.convexity())
    print("Radial variance:", rock_geometry.radial_variance())
    print("Area Sphericity:", rock_geometry.area_sphericity())
    print("Diameter Sphericity:", rock_geometry.diameter_sphericity())
    print("Circle Ratio Sphericity:", rock_geometry.circle_ratio_sphericity())
    print("Surface Area Sphericity:", rock_geometry.surface_area_sphericity())
    print("Aspect Ratio:", rock_geometry.aspect_ratio())
    drawn_image = rock_geometry.draw_all_contours(rock_geometry.image)
    # cv2.imshow('Contour Drawn', drawn_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("----------------------------------------------------")
def histstat(img):
    hist=np.zeros(8)
    print("Image:", img)
    print("Image Path:", os.path.join(path, dir, l, f))
    # rock_geometry = TwoDctRockgeometry(os.path.join(path, dir, l, f))
    rock_geometry=img
    hist[0]=(rock_geometry.convexity())
    hist[1]=(rock_geometry.circularity())
    hist[2]=(rock_geometry.radial_variance())
    hist[3]=(rock_geometry.area_sphericity())
    hist[4]=(rock_geometry.diameter_sphericity())
    hist[5]=(rock_geometry.circle_ratio_sphericity())
    hist[6]=(rock_geometry.surface_area_sphericity())
    hist[7]=(rock_geometry.aspect_ratio())

    return hist
def Atributos_histogramas(imagemsRGB):
        estatisticas=np.zeros((len(imagemsRGB),8))
        for i in range(len(imagemsRGB)):
            img=imagemsRGB[i]
            data=histstat(img)
            estatisticas[i]=data
        return estatisticas
# list_imagestain=[]
# list_imagestest=[]
# list_dir = os.listdir(path)
#         #Redimensionamento dos arrays
# list_imagestrain=[]
# list_imagestest=[]
# list_dir = os.listdir(path)

# for dir in list_dir:
#     list_labels = os.listdir(os.path.join(path,dir))
#     for l in list_labels:
#             list_files = os.listdir(os.path.join(path,dir,l))
#             for f in list_files:
#                 imgarray = mpimg.imread(os.path.join(path,dir,l,f))
#             #As dimensoes das imagems agora sao de 14x14
#                 imgarray14x14=np.resize(imgarray,(14,14))
#                 img = Image.fromarray(imgarray)
#             if dir == 'train':
#                 list_imagestrain.append(img)
#             else:
#                 list_imagestest.append(img)
path="/Users/victorvillegassalabarria/Documents/Mastria/FEM2024/TEXTODISSERTAÇÅO/ShapeDescriptors/VugFracture/"
list_dir = os.listdir(path)
list_dir.remove('.DS_Store')

# list_dir.remove('.DS_Store')
print(list_dir)
images_train = []
labels_train = []
images_test = []
labels_test = []
for dir in list_dir:
  list_labels = os.listdir(os.path.join(path,dir))
  list_labels.remove('.DS_Store')
  for l in list_labels:
    list_files = os.listdir(os.path.join(path,dir,l))
    for f in list_files:
      if list_files[0] == '.DS_Store':
        list_files.remove('.DS_Store')
      if f != '.DS_Store':
        img = TwoDctRockgeometry(os.path.join(path,dir,l,f))

        if dir == 'Train':
            images_train.append(img)
            labels_train.append(int(l))
        else:
            images_test.append(img)
            labels_test.append(int(l))

print(f'labels_train: \n{labels_train}')
print('Número de imagens de treino:', len(labels_train))

print(f'labels_test: \n{labels_test}')
print('Número de imagens de teste:', len(labels_test))
images_train1 = [rock_geometry.get_image() for rock_geometry in images_train]
images_train1[random.randint(0,150)]
images_train1[random.randint(0,10)]


n_imgs = 4
fig, axs = plt.subplots(2,n_imgs,figsize=(8, 4))

for n in range(n_imgs):
    plt.subplot(2,n_imgs,n+1)
    plt.axis('off')
    tmp = random.randint(0,150)
    plt.imshow(images_train1[tmp])
    plt.title(f'Classe {labels_train[tmp]}')

    plt.subplot(2,n_imgs,n+n_imgs+1)
    plt.axis('off')
    tmp = random.randint(0,10)
    plt.imshow(images_train1[tmp])
    plt.title(f'Classe {labels_train[tmp]}')
list_imagestrain=images_train
list_imagestest=images_test
train_features=Atributos_histogramas(list_imagestrain)
test_features=Atributos_histogramas(list_imagestest)
print(train_features.shape)
print(test_features.shape)
train_image_index = 0  # Index of the train image you want to print the parameters for
train_image_parameters = train_features[train_image_index][:8]
print("Train Image Parameters:", train_image_parameters)
###
###
#KNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
###
###
modelHistogramas = KNeighborsClassifier(n_neighbors=5)
Yhist = np.concatenate((np.zeros(85), np.ones(85))).astype(int) # rótulos das classes (1 e 0)
print(Yhist)
modelHistogramas.fit(train_features,Yhist)
Ypredtest=modelHistogramas.predict(test_features)
Ypredtreino=modelHistogramas.predict(train_features)
print("Metricas para o conjunto de datos de treinamento:\n")
print(Ypredtest)
print(Ypredtreino)
#testar o modelo
Yrealtreino =Yhist # valores reais do treino (1 e 0)
Yrealtest = np.concatenate((np.zeros(14), np.ones(14))).astype(int) # valores reais do test (1 e 0)
print("Metricas para o conjunto de datos de treinamento:\n")
acc = accuracy_score(Yrealtreino, Ypredtreino)
prec = precision_score(Yrealtreino, Ypredtreino)
rec = recall_score(Yrealtreino, Ypredtreino)
f1 = f1_score(Yrealtreino, Ypredtreino)
mconftreino=confusion_matrix(Yrealtreino, Ypredtreino)
# Imprimindo as métricas
print('Acurácia: %.2f' % acc)
print('Precisão: %.2f' % prec)
print('Recall: %.2f' % rec)
print('F1-Score: %.2f' % f1)
plt.figure(figsize=(3, 3))
sns.heatmap(mconftreino, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Clases Predichas')
plt.ylabel('Clases Verdaderas')
plt.title('Matriz de Confusión')
plt.show()
# Avaliando o modelo (as funções foram importadas previamente da biblioteca sklearn.metrics)
print("\nMetricas para o teste:\n")
acc = accuracy_score(Yrealtest, Ypredtest)
prec = precision_score(Yrealtest, Ypredtest)
rec = recall_score(Yrealtest, Ypredtest)
f1 = f1_score(Yrealtest, Ypredtest)
mconftest=confusion_matrix(Yrealtest, Ypredtest)
# Imprimindo as métricas
print('Acurácia: %.2f' % acc)
print('Precisão: %.2f' % prec)
print('Recall: %.2f' % rec)
print('F1-Score: %.2f' % f1)
plt.figure(figsize=(3, 3))
sns.heatmap(mconftest, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Clases Predichas')
plt.ylabel('Clases Verdaderas')
plt.title('Matriz de Confusión')
plt.show()
Xmedia_filas = np.mean(test_features, axis=1)
# print(Xmedia_filas)
# print(Yvarianza_filas)
Yvarianza_filas = np.var(test_features, axis=1)
plt.scatter(train_features[:,0], train_features[:,2], c=Ypredtreino)
plt.scatter(test_features[:,0], test_features[:,2], marker='x', c=Ypredtest)
plt.title('Classificação com KNN')
#plt.xlim(50, 450)
#plt.ylim(-500000, np.max(Yvarianza_filas))
plt.xlabel('Média')
plt.ylabel('Variância')
plt.show()