"""
Cálculo de IoU (Intersection over Union) entre bounding boxes
"""
import torch
from torch.ops.vision import roi_align
from torchvision.ops import box_iou
from .logger_config import get_logger
from .exceptions import ImageProcessingError

logger = get_logger(__name__)


def calculate_iou(bbox1: list, bbox2: list) -> float:
    """
    Calcula IoU entre dos bounding boxes
    
    Args:
        bbox1: [x1, y1, x2, y2] en formato xyxy
        bbox2: [x1, y1, x2, y2] en formato xyxy
    
    Returns:
        Valor de IoU entre 0 y 1
    
    Raises:
        ImageProcessingError si hay error en el cálculo
    """
    try:
        # Convertir a tensor
        boxes = torch.tensor([bbox1, bbox2], dtype=torch.float32)
        
        # Calcular matriz de IoU
        iou_matrix = box_iou(boxes, boxes)
        
        # El valor en [0,1] es el IoU que nos interesa
        iou_value = iou_matrix[0, 1].item()
        
        logger.debug(f"   IoU calculado: {iou_value:.4f}")
        
        return iou_value
    
    except Exception as e:
        raise ImageProcessingError(f"Error calculando IoU: {e}")


def calculate_iou_batch(bboxes_list1: list, bboxes_list2: list) -> torch.Tensor:
    """
    Calcula matriz de IoU entre dos listas de bboxes
    
    Args:
        bboxes_list1: Lista de bboxes [[x1,y1,x2,y2], ...]
        bboxes_list2: Lista de bboxes [[x1,y1,x2,y2], ...]
    
    Returns:
        Matriz de IoU (n x m)
    
    Raises:
        ImageProcessingError si hay error
    """
    try:
        boxes1 = torch.tensor(bboxes_list1, dtype=torch.float32)
        boxes2 = torch.tensor(bboxes_list2, dtype=torch.float32)
        
        iou_matrix = box_iou(boxes1, boxes2)
        
        return iou_matrix
    
    except Exception as e:
        raise ImageProcessingError(f"Error calculando matriz de IoU: {e}")
