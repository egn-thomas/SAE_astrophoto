import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from astropy.io import fits
from model import TinyUNet, Model
import os
import cv2 as cv

class StarDataset(Dataset):
    def __init__(self, data_dir):
        # Convertir en chemin absolu pour être sûr
        self.abs_path = os.path.abspath(data_dir)
        print(f"--- Diagnostic du dossier ---")
        print(f"Recherche dans : {self.abs_path}")
        
        if not os.path.exists(self.abs_path):
            print(f"ERREUR : Le dossier n'existe pas !")
            self.files = []
        else:
            # Lister TOUS les fichiers pour voir ce qu'il y a dedans
            all_files = os.listdir(self.abs_path)
            print(f"Fichiers totaux trouvés dans le dossier : {len(all_files)}")
            
            # Filtrer par extensions .fits et .png
            self.files = []
            for f in all_files:
                if f.lower().endswith(('.fits', '.png', '.jpg', '.jpeg')):
                    self.files.append(os.path.join(self.abs_path, f))
            
            print(f"Fichiers image détectés : {len(self.files)}")
            if len(self.files) > 0:
                print(f"Exemple de fichier : {self.files[0]}")
        print(f"-----------------------------")
        
        self.base_model = Model()
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Debug: Afficher le fichier en cours de traitement
        print(f"[DEBUG] Traitement du fichier {idx+1}/{len(self.files)}: {os.path.basename(file_path)}")
        
        # Charger l'image selon le format
        if file_path.lower().endswith('.fits'):
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                print(f"[DEBUG] FITS chargé - Shape: {data.shape}, Type: {data.dtype}, Min: {data.min():.2f}, Max: {data.max():.2f}")
        else:
            # Pour PNG/JPG, utiliser OpenCV
            data = cv.imread(file_path, cv.IMREAD_UNCHANGED)
            if data is None:
                raise ValueError(f"Impossible de charger l'image : {file_path}")
            
            # Convertir en float32
            data = data.astype(np.float32)
            print(f"[DEBUG] Image chargée - Shape: {data.shape}, Type: {data.dtype}, Min: {data.min():.2f}, Max: {data.max():.2f}")
            
            # Si c'est une image couleur, convertir en niveaux de gris
            if data.ndim == 3:
                if data.shape[2] == 3:  # RGB
                    data = cv.cvtColor(data, cv.COLOR_BGR2GRAY).astype(np.float32)
                elif data.shape[2] == 4:  # RGBA
                    data = cv.cvtColor(data, cv.COLOR_BGRA2GRAY).astype(np.float32)
                else:
                    data = np.mean(data, axis=2)  # Moyenne des canaux
                print(f"[DEBUG] Convertie en niveaux de gris - Shape: {data.shape}")
        
        if file_path.lower().endswith('.fits') and data.ndim == 3:
            if data.shape[0] == 3: # Format [C, H, W]
                data_for_detect = np.mean(data, axis=0)
            else: # Format [H, W, C]
                data_for_detect = np.mean(data, axis=2)
        else:
            data_for_detect = data

        # 1. Génération du Masque Label via ton algo actuel
        print(f"[DEBUG] Génération du masque avec detect_stars...")
        mask = self.base_model.detect_stars(data_for_detect)
        mask = (mask > 0).astype(np.float32)
        
        # Debug: Statistiques du masque
        num_stars = int(mask.sum())
        print(f"[DEBUG] Masque généré - Étoiles détectées: {num_stars}, Pourcentage couvert: {mask.mean()*100:.2f}%")
        
        # 2. Préparation de l'entrée (Normalisation améliorée pour préserver les détails)
        # Utiliser une normalisation percentile pour éviter la saturation
        p_low, p_high = np.percentile(data_for_detect, [1, 99])  # Éviter les outliers
        img_norm = np.clip(data_for_detect, p_low, p_high)
        img_norm = (img_norm - p_low) / (p_high - p_low + 1e-8)
        img_norm = img_norm.astype(np.float32)
        
        print(f"[DEBUG] Normalisation - Percentiles [1%, 99%]: [{p_low:.2f}, {p_high:.2f}], Image norm Min: {img_norm.min():.3f}, Max: {img_norm.max():.3f}")
        
        # On force une taille fixe (ex: 512x512) pour que le stack fonctionne
        target_size = (512, 512) 
        img_norm = cv.resize(img_norm, target_size, interpolation=cv.INTER_AREA)
        mask = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
        
        print(f"[DEBUG] Redimensionnement à {target_size}")
        # --------------------------------------

        input_tensor = torch.from_numpy(img_norm).unsqueeze(0) # [1, 512, 512]
        target_tensor = torch.from_numpy(mask).unsqueeze(0)   # [1, 512, 512]
        
        print(f"[DEBUG] Tensors créés - Input shape: {input_tensor.shape}, Target shape: {target_tensor.shape}")
        
        return input_tensor, target_tensor

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEBUG] Entraînement sur : {device}")

    model = TinyUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy pour masque 0/1

    dataset = StarDataset('gui/data') 
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"[DEBUG] Dataset créé - {len(dataset)} images, Batch size: 4, Nombre de batches par epoch: {len(dataloader)}")

    model.train()
    for epoch in range(20):
        print(f"\n[DEBUG] === EPOCH {epoch+1}/20 ===")
        total_loss = 0
        batch_count = 0
        
        for batch_in, batch_target in dataloader:
            batch_count += 1
            print(f"[DEBUG] Traitement du batch {batch_count}/{len(dataloader)}")
            
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            # Debug: Statistiques des données d'entrée
            print(f"[DEBUG] Input batch - Shape: {batch_in.shape}, Min: {batch_in.min().item():.3f}, Max: {batch_in.max().item():.3f}, Mean: {batch_in.mean().item():.3f}")
            print(f"[DEBUG] Target batch - Shape: {batch_target.shape}, Stars in batch: {batch_target.sum().item():.0f}, Coverage: {batch_target.mean().item()*100:.2f}%")
            
            optimizer.zero_grad()
            outputs = model(batch_in)
            
            # Debug: Statistiques des prédictions
            pred_mean = outputs.mean().item()
            pred_std = outputs.std().item()
            print(f"[DEBUG] Predictions - Mean: {pred_mean:.3f}, Std: {pred_std:.3f}, Min: {outputs.min().item():.3f}, Max: {outputs.max().item():.3f}")
            
            loss = criterion(outputs, batch_target)
            current_loss = loss.item()
            print(f"[DEBUG] Loss du batch: {current_loss:.4f}")
            
            loss.backward()
            optimizer.step()
            
            total_loss += current_loss
        
        avg_loss = total_loss / len(dataloader)
        print(f"[DEBUG] Epoch {epoch+1}/20 terminé - Loss moyenne: {avg_loss:.4f}")

    # Sauvegarde finale
    torch.save(model.state_dict(), 'star_unet.pth')
    print("[DEBUG] Modèle sauvegardé sous 'star_unet.pth'")

if __name__ == "__main__":
    train()