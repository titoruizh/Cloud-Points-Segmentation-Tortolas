import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast as autocast_legacy 
import yaml
import argparse
import wandb
import os
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Para evitar issues sin display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.utils.metrics import IoUCalculator 

def create_iou_visualization(xyz_sample, labels_sample, preds_sample, iou_maq, iou_suelo, epoch):
    """
    Crea una visualización comparativa de ground truth vs predicciones
    enfocada en las zonas donde hay maquinaria.
    
    Args:
        xyz_sample: Tensor [N, 3] - coordenadas XYZ de puntos
        labels_sample: Tensor [N] - etiquetas ground truth
        preds_sample: Tensor [N] - predicciones del modelo
        iou_maq: float - IoU de maquinaria calculado
        iou_suelo: float - IoU de suelo calculado
        epoch: int - número de época actual
    
    Returns:
        matplotlib.figure.Figure - figura lista para loguear a wandb
    """
    # Convertir a numpy y mover a CPU
    xyz_np = xyz_sample.cpu().numpy()
    labels_np = labels_sample.cpu().numpy()
    preds_np = preds_sample.cpu().numpy()
    
    # Identificar dónde está la maquinaria en ground truth o predicciones
    maq_mask_gt = labels_np == 0
    maq_mask_pred = preds_np == 0
    
    # Si hay maquinaria, hacer zoom a esa zona + margen
    if np.any(maq_mask_gt | maq_mask_pred):
        # Encontrar límites de la maquinaria
        maq_points = xyz_np[maq_mask_gt | maq_mask_pred]
        
        if len(maq_points) > 10:  # Mínimo 10 puntos para hacer zoom
            x_min, x_max = maq_points[:, 0].min(), maq_points[:, 0].max()
            y_min, y_max = maq_points[:, 1].min(), maq_points[:, 1].max()
            z_min, z_max = maq_points[:, 2].min(), maq_points[:, 2].max()
            
            # Agregar margen del 50% para contexto (mínimo 1 metro)
            margin_x = max((x_max - x_min) * 0.5, 1.0)
            margin_y = max((y_max - y_min) * 0.5, 1.0)
            margin_z = max((z_max - z_min) * 0.5, 0.5)
            
            # Filtrar puntos en la región de interés
            roi_mask = (
                (xyz_np[:, 0] >= x_min - margin_x) & (xyz_np[:, 0] <= x_max + margin_x) &
                (xyz_np[:, 1] >= y_min - margin_y) & (xyz_np[:, 1] <= y_max + margin_y) &
                (xyz_np[:, 2] >= z_min - margin_z) & (xyz_np[:, 2] <= z_max + margin_z)
            )
            
            # Solo aplicar filtro si quedan suficientes puntos
            if np.sum(roi_mask) > 100:
                xyz_np = xyz_np[roi_mask]
                labels_np = labels_np[roi_mask]
                preds_np = preds_np[roi_mask]
    
    # Limitar puntos para claridad visual
    max_points = 15000
    if len(xyz_np) > max_points:
        indices = np.random.choice(len(xyz_np), max_points, replace=False)
        xyz_np = xyz_np[indices]
        labels_np = labels_np[indices]
        preds_np = preds_np[indices]
    
    # Definir colores: Maquinaria (rojo brillante), Suelo (gris claro)
    cmap = ListedColormap(['#FF3030', '#A0A0A0'])  # Rojo para Maq (0), Gris para Suelo (1)
    
    # Crear figura con vista aérea y lateral
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # FILA 1: Vista Aérea (X-Y)
    # Ground Truth
    scatter1 = axes[0, 0].scatter(xyz_np[:, 0], xyz_np[:, 1], c=labels_np, 
                                  cmap=cmap, s=3, alpha=0.8, vmin=0, vmax=1)
    axes[0, 0].set_title(f'Ground Truth - Vista Aérea (Época {epoch})', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Predicciones
    scatter2 = axes[0, 1].scatter(xyz_np[:, 0], xyz_np[:, 1], c=preds_np, 
                                  cmap=cmap, s=3, alpha=0.8, vmin=0, vmax=1)
    axes[0, 1].set_title(f'Predicciones - Vista Aérea | IoU Maq: {iou_maq:.2f}%', 
                         fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FILA 2: Vista Lateral (X-Z)
    # Ground Truth
    scatter3 = axes[1, 0].scatter(xyz_np[:, 0], xyz_np[:, 2], c=labels_np, 
                                  cmap=cmap, s=3, alpha=0.8, vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth - Vista Lateral', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Z (altura, m)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Predicciones
    scatter4 = axes[1, 1].scatter(xyz_np[:, 0], xyz_np[:, 2], c=preds_np, 
                                  cmap=cmap, s=3, alpha=0.8, vmin=0, vmax=1)
    axes[1, 1].set_title(f'Predicciones - Vista Lateral | IoU Suelo: {iou_suelo:.2f}%', 
                         fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Z (altura, m)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Agregar colorbar compartido
    cbar = plt.colorbar(scatter2, ax=axes.ravel().tolist(), ticks=[0, 1], fraction=0.02, pad=0.04)
    cbar.ax.set_yticklabels(['Maquinaria', 'Suelo'])
    
    plt.tight_layout()
    
    return fig

def get_config():
    parser = argparse.ArgumentParser(description='Entrenamiento Point Cloud Research')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo YAML de configuración')
    
    # Permitir argumentos extra para Sweeps (ej: --train.learning_rate=0.001)
    args, unknown_args = parser.parse_known_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # --- 🔧 CLI OVERRIDES AUTOMÁTICOS ---
    # Parsea argumentos estilo --seccion.clave=valor y actualiza el dict de config
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key_full = arg[2:] # Quitar --
            key_path, val_str = key_full.split("=", 1)
            
            # Intentar parsear el valor (lista, int, float) con YAML
            try:
                parsed_val = yaml.safe_load(val_str)
            except:
                parsed_val = val_str
            
            # Navegar y actualizar config
            keys = key_path.split('.')
            current = config
            try:
                for k in keys[:-1]:
                    if k not in current: 
                        current[k] = {}
                    current = current[k]
                
                # Actualizar hoja
                current[keys[-1]] = parsed_val
                # print(f"🔧 Override CLI: {key_path} = {parsed_val}")
            except Exception as e:
                print(f"⚠️ Error aplicando override {key_path}: {e}")
                
    return config

def main():
    # 1. Cargar Configuración Base (YAML)
    cfg = get_config()
    exp_name = cfg['experiment']['name']
    
    # 2. Inicializar W&B con grupo
    run = wandb.init(
        entity=cfg['experiment']['entity'],
        project=cfg['experiment']['project'],
        name=exp_name,
        group=cfg['experiment'].get('group', exp_name),  # ✅ Usar grupo del config
        job_type="train",
        config=cfg,
        settings=wandb.Settings(start_method="fork")
    )
    
    # ============================================================
    # ### BLOQUE SWEEP: INTERCEPTAR Y SOBREESCRIBIR (PRIMERO) ###
    # ============================================================
    # Primero actualizamos la configuración con lo que diga el Sweep
    
    if wandb.config.get('learning_rate'):
        new_lr = wandb.config.learning_rate
        cfg['train']['learning_rate'] = new_lr
        print(f"🧹 SWEEP: Learning Rate actualizado a {new_lr}")

    if wandb.config.get('weight_maq'):
        w_maq = wandb.config.weight_maq
        # Mapping: 0=Suelo, 1=Maquinaria. Queremos penalizar error en Maq.
        cfg['data']['class_weights'] = [1.0, float(w_maq)] 
        print(f"🧹 SWEEP: Peso Maquinaria actualizado a {w_maq}")

    if wandb.config.get('jitter_sigma'):
        new_sigma = wandb.config.jitter_sigma
        cfg['augmentation']['jitter_sigma'] = new_sigma
        print(f"🧹 SWEEP: Jitter Sigma actualizado a {new_sigma}")
        
    if wandb.config.get('base_radius'):
        new_radius = wandb.config.base_radius
        cfg['model']['base_radius'] = new_radius
        print(f"🧹 SWEEP: Radio Base actualizado a {new_radius}")

    if wandb.config.get('epochs'):
        new_epochs = wandb.config.epochs
        cfg['train']['epochs'] = new_epochs
        print(f"🧹 SWEEP TEST: Épocas forzadas a {new_epochs}")

    # ============================================================
    # ### RENOMBRAMIENTO INTELIGENTE (DESPUÉS) ###
    # ============================================================
    # Ahora que 'cfg' tiene los valores reales del Sweep, generamos el nombre.
    
    if wandb.run.sweep_id:
        lr = cfg['train']['learning_rate']
        # Usamos el peso de la clase 1 (Maquinaria) para el nombre
        w = int(cfg['data']['class_weights'][1]) 
        jit = cfg['augmentation']['jitter_sigma']
        
        # Para PointNet2, incluir base_radius en el nombre
        arch = cfg['model']['architecture']
        if arch == "PointNet2" and 'base_radius' in cfg['model']:
            radius = cfg['model']['base_radius']
            run_id_tag = f"LR{lr:.4f}_W{w}_J{jit:.3f}_R{radius:.1f}"
        else:
            run_id_tag = f"LR{lr:.4f}_W{w}_J{jit:.3f}"
        
        # Actualizamos el nombre en la nube
        wandb.run.name = run_id_tag
        print(f"🏷️ Run bautizado como: {run_id_tag}")
    else:
        run_id_tag = wandb.run.name if wandb.run.name else "Manual"
    # ============================================================
    
    device = torch.device(cfg['system']['device'])
    print(f"🚀 Iniciando entrenamiento: {exp_name}")
    print(f"   ID de W&B: {run.id}")
    print(f"   Hardware: {device}")

    # Crear directorio de checkpoints organizado
    if wandb.run.sweep_id:
        # Si es sweep, crear carpeta específica
        checkpoint_dir = os.path.join("checkpoints", f"SWEEP_{exp_name}")
    else:
        # Si es manual, usar carpeta raíz
        checkpoint_dir = "checkpoints"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Checkpoints se guardarán en: {checkpoint_dir}")

    # 3. Preparar Datos
    print("📂 Cargando Dataset...")
    
    # SELECCIÓN DINÁMICA DE DATASET (V3 vs V4)
    d_in = cfg['model'].get('d_in', 6)
    
    if d_in == 10:
        print("🌈 Detectado V4 (RGB Mode, d_in=10). Usando src.data.dataset_v4")
        from src.data.dataset_v4 import MiningDataset
    elif d_in == 9:
        if "V6" in cfg['data']['path']:
             print("🧪 Detectado V6 (Resolution Sync, 0.25m). Usando src.data.dataset_v6")
             from src.data.dataset_v6 import MiningDataset
        else:
             print("🧪 Detectado V5 (RGB No-Vert, d_in=9). Usando src.data.dataset_v5")
             from src.data.dataset_v5 import MiningDataset
    else:
        print("🛡️ Detectado V3 Legacy (d_in<9). Usando src.data.dataset_v3")
        from src.data.dataset_v3 import MiningDataset

    full_dataset = MiningDataset(
        data_dir=cfg['data']['path'], 
        num_points=cfg['data']['num_points'], 
        split='train',
        aug_config=cfg['augmentation'],
        oversample_machinery=cfg['data'].get('oversample_machinery', 0)
    )
    
    train_size = int(cfg['data']['train_split_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"📊 Datos cargados: {len(full_dataset)} total | {train_size} Train | {val_size} Val")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['system']['num_workers'],
        pin_memory=cfg['system'].get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['system']['num_workers'],
        pin_memory=cfg['system'].get('pin_memory', True)
    )

    # 4. Inicializar Modelo
    arch_name = cfg['model']['architecture']
    num_classes = cfg['model'].get('num_classes', 2)
    d_in = cfg['model'].get('d_in', 6)

    print(f"🏗️ Cargando arquitectura: {arch_name}")

    if arch_name == "RandLANet":
        from src.models.randlanet import RandLANet
        model = RandLANet(d_in=d_in, num_classes=num_classes).to(device)
    elif arch_name == "MiniPointNet":
        from src.models.minipointnet import MiniPointNet 
        model = MiniPointNet(d_in=d_in, num_classes=num_classes).to(device)
    elif arch_name == "PointNet2":
        from src.models.pointnet2 import PointNet2 
        radius = cfg['model'].get('base_radius', 1.0)
        print(f"   📡 Radio Base configurado: {radius}m")
        model = PointNet2(d_in=d_in, num_classes=num_classes, base_radius=radius).to(device)
    else:
        raise ValueError(f"❌ Arquitectura desconocida: {arch_name}. Revisa tu YAML.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    
    scheduler = None
    if cfg['train'].get('scheduler') == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg['train']['epochs'], 
            eta_min=cfg['train'].get('scheduler_min_lr', 0.0001)
        )
    elif cfg['train'].get('scheduler') == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['train'].get('scheduler_step_size', 15),
            gamma=cfg['train'].get('scheduler_gamma', 0.5)
        )
        
    weights = torch.tensor(cfg['data']['class_weights']).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    scaler = torch.amp.GradScaler('cuda')
    wandb.watch(model, log="all", log_freq=100)
    
    best_val_loss = float('inf')
    best_iou_maq = 0.0  # Empezamos en 0
    
    accumulations = cfg['data'].get('accumulations', 1) 
    label_offset = cfg['data'].get('label_offset', 0)
    print(f"ℹ️ Label Offset: -{label_offset}")

    # --- BUCLE ---
    for epoch in range(cfg['train']['epochs']):
        
        # === ENTRENAMIENTO CON MÉTRICAS ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total_points = 0
        train_iou_calc = IoUCalculator(num_classes=num_classes)
        
        optimizer.zero_grad()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [Train]")
        
        for i, (xyz, features, labels) in enumerate(loop):
            xyz, features, labels = xyz.to(device), features.to(device), labels.to(device)
            labels = torch.clamp(labels - label_offset, 0, 1)
            
            with torch.amp.autocast('cuda'):
                outputs = model(xyz, features)
                loss = criterion(outputs, labels)
                loss = loss / accumulations 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulations == 0:
                # --- GRADIENT CLIPPING (Anti-NaN) ---
                scaler.unscale_(optimizer) # Desescalar antes de clippear
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_loss = loss.item() * accumulations
            train_loss += current_loss
            
            # Calcular métricas de train (sin gradientes)
            with torch.no_grad():
                _, preds = torch.max(outputs.data, 1)
                train_correct += (preds == labels).sum().item()
                train_total_points += labels.numel()
                train_iou_calc.add_batch(preds, labels)
            
            loop.set_postfix(loss=current_loss)
            
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total_points
        
        # Calcular IoU de entrenamiento
        train_iou_per_class, train_miou = train_iou_calc.compute_iou()
        train_iou_suelo = train_iou_per_class[0] * 100 if len(train_iou_per_class) > 0 else 0
        train_iou_maq = train_iou_per_class[1] * 100 if len(train_iou_per_class) > 1 else 0
        train_miou = train_miou * 100
        
        # === VALIDACIÓN ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total_points = 0
        val_iou_calc = IoUCalculator(num_classes=num_classes)
        
        # Variables para guardar una muestra para visualización
        viz_xyz = None
        viz_labels = None
        viz_preds = None
        
        with torch.no_grad():
            for batch_idx, (xyz, features, labels) in enumerate(val_loader):
                xyz = xyz.to(device)
                features = features.to(device)
                labels = labels.to(device)
                labels = torch.clamp(labels - label_offset, 0, 1)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(xyz, features)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(outputs.data, 1)
                val_correct += (preds == labels).sum().item()
                val_total_points += labels.numel()
                val_iou_calc.add_batch(preds, labels)
                
                # Selección Inteligente de Visualización:
                # Priorizar bloques que tengan Maquinaria (Clase 1) para no ver solo suelo
                if viz_xyz is None: # Si aún no tenemos nada, guardamos el primero por si acaso
                     viz_xyz = xyz[0]
                     viz_labels = labels[0]
                     viz_preds = preds[0]
                
                # Buscar en el batch actual un ejemplo MEJOR (con maquinaria)
                # Iterar sobre el batch
                for b in range(xyz.shape[0]):
                    if (labels[b] == 1).sum() > 50: # Si tiene más de 50 puntos de maquinara
                        viz_xyz = xyz[b]
                        viz_labels = labels[b]
                        viz_preds = preds[b]
                        break # Encontramos uno bueno en este batch, nos quedamos con este (o seguimos buscando en siguientes batches si queremos el 'mejor' absoluto, pero con esto basta)
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total_points
        
        # Calcular IoU de validación
        val_iou_per_class, val_miou = val_iou_calc.compute_iou()
        val_iou_suelo = val_iou_per_class[0] * 100 if len(val_iou_per_class) > 0 else 0
        val_iou_maq = val_iou_per_class[1] * 100 if len(val_iou_per_class) > 1 else 0
        val_miou = val_miou * 100
        
        current_lr = cfg['train']['learning_rate']
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        # Imprimir métricas
        print(f"\n   📊 TRAIN  → Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.2f}% | mIoU: {train_miou:.2f}%")
        print(f"      🚜 IoU Maq: {train_iou_maq:.2f}% | 🟤 IoU Suelo: {train_iou_suelo:.2f}%")
        print(f"   📊 VAL    → Loss: {avg_val_loss:.4f} | Acc: {val_accuracy:.2f}% | mIoU: {val_miou:.2f}%")
        print(f"      🚜 IoU Maq: {val_iou_maq:.2f}% | 🟤 IoU Suelo: {val_iou_suelo:.2f}%")
        
        # Crear visualización del IoU de maquinaria (SOLO cada 10 épocas o la primera)
        # Crear visualización del IoU de maquinaria (SOLO cada 10 épocas o la primera)
        if viz_xyz is not None and ((epoch + 1) % 10 == 0 or epoch == 0):
            try:
                # 1. Imagen 2D (Matplotlib) para overview rápido
                viz_fig = create_iou_visualization(
                    viz_xyz, viz_labels, viz_preds, 
                    val_iou_maq, val_iou_suelo, epoch + 1
                )
                wandb_image = wandb.Image(viz_fig, caption=f"Epoch {epoch+1} 2D")
                plt.close(viz_fig)

                # 2. Nube 3D Interactiva (WandB Object3D)
                # Preparamos datos: [x, y, z, r, g, b] (o labels)
                # WandB espera numpy [N, 3] o [N, 4/6]
                
                # Convertir a numpy CPU
                xyz_cpu = viz_xyz.cpu().numpy()
                preds_cpu = viz_preds.cpu().numpy().reshape(-1, 1)
                labels_cpu = viz_labels.cpu().numpy().reshape(-1, 1) # GT
                
                # Puntos para Predicción (Rojo=Maq, Gris=Suelo)
                # Creamos colores RGB manualmente para wandb.Object3D
                # Maquinaria (1): [255, 0, 0], Suelo (0): [128, 128, 128]
                colors_pred = np.zeros((xyz_cpu.shape[0], 3))
                colors_pred[preds_cpu[:,0] == 1] = [255, 0, 0]    # Rojo Maquina
                colors_pred[preds_cpu[:,0] == 0] = [128, 128, 128] # Gris Suelo
                
                # Puntos para Ground Truth
                colors_gt = np.zeros((xyz_cpu.shape[0], 3))
                colors_gt[labels_cpu[:,0] == 1] = [0, 255, 0]    # Verde Maquina (GT)
                colors_gt[labels_cpu[:,0] == 0] = [128, 128, 128] # Gris Suelo

                points_pred = np.hstack((xyz_cpu, colors_pred))
                points_gt = np.hstack((xyz_cpu, colors_gt))

                wandb_3d_pred = wandb.Object3D(points_pred, caption=f"Epoch {epoch+1} Prediction (Red=Machinery)")
                wandb_3d_gt = wandb.Object3D(points_gt, caption=f"Epoch {epoch+1} Ground Truth (Green=Machinery)")
                
            except Exception as e:
                print(f"⚠️ Error creando visualización: {e}")
                wandb_image = None
                wandb_3d_pred = None
                wandb_3d_gt = None
        else:
            wandb_image = None
            wandb_3d_pred = None
            wandb_3d_gt = None
        
        # Loguear a WandB con prefijos train/val
        log_dict = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            
            # Métricas de TRAIN
            "train/loss": avg_train_loss,
            "train/accuracy": train_accuracy,
            "train/mIoU": train_miou,
            "train/IoU_Maquinaria": train_iou_maq,
            "train/IoU_Suelo": train_iou_suelo,
            
            # Métricas de VAL
            "val/loss": avg_val_loss,
            "val/accuracy": val_accuracy,
            "val/mIoU": val_miou,
            "val/IoU_Maquinaria": val_iou_maq,
            "val/IoU_Suelo": val_iou_suelo,
        }

        # Solo agregar imagen si corresponde
        if wandb_image:
            log_dict["val/visualization"] = wandb_image
        
        if wandb_3d_pred:
            log_dict["val/3D_Prediction"] = wandb_3d_pred
            log_dict["val/3D_GroundTruth"] = wandb_3d_gt
            
        wandb.log(log_dict)

        
        # ============================================================
        # === ESTRATEGIA DE GUARDADO DOBLE (Solo los Mejores) ===
        # ============================================================
        
        # Generamos el nombre base limpio
        safe_tag = run_id_tag.replace("/", "-") 
        
        # 1. GUARDADO "BEST LOSS" (El Matemático - Estabilidad)
        if avg_val_loss < best_val_loss:
            print(f"   📉 ¡Récord Matemático! Val Loss bajó de {best_val_loss:.4f} a {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            
            # Nombre específico: _BEST_LOSS
            nombre_loss = f"{safe_tag}_BEST_LOSS.pth"
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, nombre_loss))
            
            wandb.run.summary["best_val_loss"] = best_val_loss

        # 2. GUARDADO "BEST IOU MAQUINARIA" (El Detective - PRIORIDAD 1)
        if val_iou_maq > best_iou_maq:
            print(f"   🔥 ¡Récord de Calidad! IoU Maq (val) subió de {best_iou_maq:.2f}% a {val_iou_maq:.2f}%")
            best_iou_maq = val_iou_maq
            
            # Nombre específico: _BEST_IOU
            nombre_iou = f"{safe_tag}_BEST_IOU.pth"
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, nombre_iou))
            
            wandb.run.summary["best_iou_maq"] = best_iou_maq

    # Fin del bucle for epoch
    print("🏁 Entrenamiento finalizado.")
    wandb.finish()

if __name__ == '__main__':
    main()