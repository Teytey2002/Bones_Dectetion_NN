
#------------------- Add for test bounding boxes on images -------------------#

#
## Charger une image du dataset
#image, label = train_dataset[0]
#
## Affichage avant transformation
#plt.imshow(TF.to_pil_image(image))
#plt.title("Image après transformation")
#plt.axis("off")
#plt.show()
#
#
#def denormalize(tensor):
#    return tensor * 0.5 + 0.5  # Inversion de la normalisation
#
#image_tensor = denormalize(image)  # Appliquer avant affichage
#plt.imshow(image_tensor.permute(1, 2, 0).numpy())  # Convertir pour affichage
#plt.title("Image après denormalisation")
#plt.axis("off")
#plt.show()
#
#
#
## Add for test bounding boxes on images
#
## Fonction pour afficher une image avec le polygone associé
#def show_image_with_polygon(image, label, class_names):
#    image = image.permute(1, 2, 0).numpy()  # Permuter les axes pour l'affichage
#    fig, ax = plt.subplots(1)
#    ax.imshow(image)
#
#    class_id = int(label[0])  # Extraire la classe
#    coords = label[1:].reshape(-1, 2)  # Reshape en (N, 2) pour avoir les points (x, y)
#    
#    # Convertir les coordonnées normalisées en pixels
#    width, height = image.shape[1], image.shape[0]
#    polygon_points = [(x * width, y * height) for x, y in coords]
#
#    # Ajout du polygone
#    polygon = patches.Polygon(polygon_points, linewidth=2, edgecolor='r', facecolor='none')
#    ax.add_patch(polygon)
#
#    # Ajouter le texte de la classe en haut à droite
#    ax.text(width - 10, 10, class_names[class_id], 
#            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5), ha='right', va='top')
#
#    plt.axis("off")
#    plt.show()
#
## Liste des noms des classes (correspondant au dataset YOLO)
#class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 
#               'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
#
## Afficher quelques images du batch avec leurs polygones
#for i in range(5):  # Afficher 5 images
#    show_image_with_polygon(images[i], labels[i], class_names)
#___________________________________________________________________________________________________________________________




#-------------------------------------- test impact resize --------------------------------------#
## Vérification du resize sans transformation
#image, label = train_dataset[110]
#
## Affichage avant resize
#original_image = Image.open(os.path.join(data_dir_train, "images", train_dataset.image_filenames[110])).convert("RGB")
#
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(original_image)
#ax[0].set_title("Image Originale")
#
## Convertir les coordonnées YOLO en pixels pour l'image originale
#original_width, original_height = original_image.size
#coords_original = label[1:].reshape(-1, 2) * torch.tensor([original_width, original_height])
#polygon_original = patches.Polygon(coords_original.numpy(), linewidth=2, edgecolor='r', facecolor='none')
#ax[0].add_patch(polygon_original)
#
## Affichage après resize
#ax[1].imshow(TF.to_pil_image(image))
#ax[1].set_title("Image Après Resize")
#
## Convertir les coordonnées YOLO en pixels pour l'image transformée
#coords_resized = label[1:].reshape(-1, 2) * torch.tensor([image_size, image_size])
#polygon_resized = patches.Polygon(coords_resized.numpy(), linewidth=2, edgecolor='r', facecolor='none')
#ax[1].add_patch(polygon_resized)
#
#plt.show()