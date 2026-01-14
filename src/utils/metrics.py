import torch
import numpy as np

class IoUCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Matriz de Confusión: Filas=Real, Columnas=Predicción
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.ulonglong)

    def add_batch(self, preds, labels):
        """
        preds: Tensor [B, N] (índices predichos 0 o 1)
        labels: Tensor [B, N] (índices reales 0 o 1)
        """
        preds = preds.detach().cpu().numpy().flatten()
        labels = labels.detach().cpu().numpy().flatten()
        
        # Truco de NumPy para calcular matriz rápida
        mask = (labels >= 0) & (labels < self.num_classes)
        
        # Calculamos los conteos
        conteos = np.bincount(
            self.num_classes * labels[mask] + preds[mask],
            minlength=self.num_classes ** 2
        )
        
        # --- FIX: CASTING EXPLÍCITO ---
        # Forzamos que 'conteos' sea del mismo tipo que la matriz (uint64)
        conteos = conteos.astype(np.ulonglong) 
        
        self.confusion_matrix += conteos.reshape(self.num_classes, self.num_classes)

    def compute_iou(self):
        """
        Retorna:
        - iou_per_class: Lista con IoU de cada clase
        - miou: Promedio
        """
        # Intersección: Diagonal de la matriz (Acertaste Clase X y era Clase X)
        intersection = np.diag(self.confusion_matrix)
        
        # Unión: (Suma Fila + Suma Columna - Intersección)
        # Real + Predicho - Lo que se cuenta doble
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        
        # Evitar división por cero
        iou = intersection / (union + 1e-10)
        miou = np.mean(iou)
        
        return iou, miou
    
    def get_confusion_matrix(self):
        return self.confusion_matrix