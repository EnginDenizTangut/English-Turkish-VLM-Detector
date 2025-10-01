import cv2
import numpy as np
from ultralytics import YOLO
import ollama
from PIL import Image
import io
import base64
import json
import os
import time
from pathlib import Path

class VLMDetector:
    def __init__(self, mode='detection'):
        """
        Initialize VLM Detector
        Args:
            mode (str): 'detection' or 'segmentation'
        """
        self.mode = mode
        if mode == 'segmentation':
            self.model = YOLO('yolov8n-seg.pt')  # Segmentation model
        else:
            self.model = YOLO('yolov8n.pt')  # Detection model
        
        self.class_names = self.model.names
        
        # Renk eşleştirmesi - Türkçe renk isimlerini RGB değerlerine çevirir
        self.color_mapping = {
            'kırmızı': (0, 0, 255),      # BGR formatında
            'kirmizi': (0, 0, 255),
            'kırmızı': (0, 0, 255),
            'mavi': (255, 0, 0),
            'yeşil': (0, 255, 0),
            'yesil': (0, 255, 0),
            'sarı': (0, 255, 255),
            'sari': (0, 255, 255),
            'mor': (255, 0, 255),
            'turuncu': (0, 165, 255),
            'pembe': (203, 192, 255),
            'siyah': (0, 0, 0),
            'beyaz': (255, 255, 255),
            'gri': (128, 128, 128),
            'kahverengi': (42, 42, 165),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'lacivert': (139, 0, 0),
            'altın': (0, 215, 255),
            'altin': (0, 215, 255),
            'gümüş': (192, 192, 192),
            'gumus': (192, 192, 192),
            'default': (0, 255, 0)  # Varsayılan yeşil
        }
    
    #TODO detect objects
    def detect_objects(self, image_path):
        results = self.model(image_path)
        return results[0]
    
    def detect_objects_direct(self, frame):
        """Direct detection on frame (faster for real-time)"""
        results = self.model(frame)
        return results[0]
    
    #TODO filterin object by classes
    def filter_objects_by_class(self, results, target_class):
        filtered_boxes = []
        filtered_confidences = []
        filtered_classes = []
        
        available_classes = list(self.class_names.values())
        
        # Renk bilgisini çıkar
        detected_color = self.extract_color_from_query(target_class)
        color_name = [name for name, value in self.color_mapping.items() if value == detected_color and name != 'default'][0]
        
        # Renk bilgisini target_class'dan çıkar
        clean_target = target_class.lower()
        for color in self.color_mapping.keys():
            if color != 'default' and color in clean_target:
                clean_target = clean_target.replace(color, '').strip()
                break
        
        llm_prompt = f"""
        Kullanıcı "{target_class}" nesnesini arıyor.
        
        Mevcut COCO sınıfları: {', '.join(available_classes)}
        
        Bu sınıflardan hangileri "{clean_target}" ile eşleşiyor? 
        
        Önemli: Türkçe kelimeleri İngilizce COCO sınıflarıyla eşleştir. Aynı anlama gelen farklı kelimeleri de düşün:
        
        İNSANLAR:
        - "insan", "kişi", "adam", "kadın", "çocuk", "bebek", "yaşlı", "genç" → "person"
        
        HAYVANLAR:
        - "kedi", "pisi", "miyav" → "cat"
        - "köpek", "it", "hav hav" → "dog" 
        - "kuş", "kanatlı" → "bird"
        - "at", "beygir" → "horse"
        - "inek", "sığır" → "cow"
        - "koyun", "kuzu" → "sheep"
        - "fil", "fildişi" → "elephant"
        - "ayı", "boz ayı" → "bear"
        - "zebra" → "zebra"
        - "zürafa", "uzun boyunlu" → "giraffe"
        - "hayvan", "canlı", "evcil hayvan" → "cat, dog, bird, horse, cow, sheep, elephant, bear, zebra, giraffe"
        
        ARAÇLAR:
        - "araba", "otomobil", "taşıt", "vasıta" → "car"
        - "kamyon", "tır", "yük aracı" → "truck"
        - "otobüs", "şehir otobüsü" → "bus"
        - "motosiklet", "moto", "motor" → "motorcycle"
        - "bisiklet", "velespit", "pedal" → "bicycle"
        - "uçak", "tayyare", "hava aracı" → "airplane"
        - "tren", "demiryolu" → "train"
        - "tekne", "gemi", "deniz aracı" → "boat"
        - "araç", "taşıt", "vasıta", "ulaşım aracı" → "car, truck, bus, motorcycle, bicycle, airplane, train, boat"
        
        YİYECEKLER:
        - "elma", "kırmızı elma", "yeşil elma" → "apple"
        - "muz", "sarı meyve" → "banana"
        - "pizza", "italyan yemeği" → "pizza"
        - "pasta", "kek", "tatlı" → "cake"
        - "donut", "halka tatlı" → "donut"
        - "sandviç", "ekmek arası" → "sandwich"
        - "portakal", "turunç" → "orange"
        - "brokoli", "yeşil sebze" → "broccoli"
        - "havuç", "turuncu sebze" → "carrot"
        - "sosisli", "hot dog" → "hot dog"
        - "yiyecek", "yemek", "besin", "gıda" → "apple, banana, pizza, cake, donut, sandwich, orange, broccoli, carrot, hot dog"
        
        EŞYALAR:
        - "sandalye", "oturak", "koltuk" → "chair"
        - "masa", "yemek masası", "çalışma masası" → "dining table"
        - "televizyon", "tv", "ekran" → "tv"
        - "laptop", "dizüstü", "bilgisayar" → "laptop"
        - "telefon", "cep telefonu", "mobil" → "cell phone"
        - "kitap", "yayın", "eser" → "book"
        - "saat", "zaman aleti" → "clock"
        - "çanta", "torba", "kese" → "handbag, backpack, suitcase"
        - "ayakkabı", "bot", "terlik" → "shoe, boot"
        - "giysi", "elbise", "kıyafet" → "clothing"
        
        ELEKTRONİK:
        - "elektronik", "teknoloji", "cihaz" → "tv, laptop, cell phone, remote, keyboard, mouse, microwave, oven, toaster, refrigerator"
        
        EV EŞYALARI:
        - "ev eşyası", "mobilya", "ev aleti" → "chair, couch, bed, dining table, toilet, tv, laptop, microwave, oven, toaster, sink, refrigerator"
        
        SADECE eşleşen COCO sınıf isimlerini virgülle ayırarak ver. Başka açıklama yapma.
        Örnek: person, car, truck
        """
        
        llm_response = self.ask_llm(llm_prompt)
        print(f"LLM sınıf eşleştirmesi: {llm_response}")
        
        lines = llm_response.strip().split('\n')
        matching_classes = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('"') and not line.startswith('Ayrıca'):
                if ',' in line:
                    classes = [cls.strip().lower() for cls in line.split(',')]
                    matching_classes.extend(classes)
                else:
                    if line.lower() in [name.lower() for name in available_classes]:
                        matching_classes.append(line.lower())
        
        if not matching_classes:
            matching_classes = [cls.strip().lower() for cls in llm_response.split(',')]
        
        print(f"Parse edilen sınıflar: {matching_classes}")
        
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])
            
            if class_name.lower() in matching_classes:
                filtered_boxes.append(box.xyxy[0].cpu().numpy())
                filtered_confidences.append(confidence)
                filtered_classes.append(class_name)
        
        return filtered_boxes, filtered_confidences, filtered_classes
    #TODO filter by class but this time for segmentation
    def filter_objects_by_class_segmentation(self, results, target_class):
        """Segmentation için sınıf bazında filtreleme"""
        filtered_masks = []
        filtered_confidences = []
        filtered_classes = []
        
        available_classes = list(self.class_names.values())
        
        # Renk bilgisini çıkar
        detected_color = self.extract_color_from_query(target_class)
        color_name = [name for name, value in self.color_mapping.items() if value == detected_color and name != 'default'][0]
        
        # Renk bilgisini target_class'dan çıkar
        clean_target = target_class.lower()
        for color in self.color_mapping.keys():
            if color != 'default' and color in clean_target:
                clean_target = clean_target.replace(color, '').strip()
                break
        
        llm_prompt = f"""
        Kullanıcı "{target_class}" nesnesini arıyor.
        
        Mevcut COCO sınıfları: {', '.join(available_classes)}
        
        Bu sınıflardan hangileri "{clean_target}" ile eşleşiyor? 
        
        Önemli: Türkçe kelimeleri İngilizce COCO sınıflarıyla eşleştir. Aynı anlama gelen farklı kelimeleri de düşün:
        
        İNSANLAR:
        - "insan", "kişi", "adam", "kadın", "çocuk", "bebek", "yaşlı", "genç" → "person"
        
        HAYVANLAR:
        - "kedi", "pisi", "miyav" → "cat"
        - "köpek", "it", "hav hav" → "dog" 
        - "kuş", "kanatlı" → "bird"
        - "at", "beygir" → "horse"
        - "inek", "sığır" → "cow"
        - "koyun", "kuzu" → "sheep"
        - "fil", "fildişi" → "elephant"
        - "ayı", "boz ayı" → "bear"
        - "zebra" → "zebra"
        - "zürafa", "uzun boyunlu" → "giraffe"
        - "hayvan", "canlı", "evcil hayvan" → "cat, dog, bird, horse, cow, sheep, elephant, bear, zebra, giraffe"
        
        ARAÇLAR:
        - "araba", "otomobil", "taşıt", "vasıta" → "car"
        - "kamyon", "tır", "yük aracı" → "truck"
        - "otobüs", "şehir otobüsü" → "bus"
        - "motosiklet", "moto", "motor" → "motorcycle"
        - "bisiklet", "velespit", "pedal" → "bicycle"
        - "uçak", "tayyare", "hava aracı" → "airplane"
        - "tren", "demiryolu" → "train"
        - "tekne", "gemi", "deniz aracı" → "boat"
        - "araç", "taşıt", "vasıta", "ulaşım aracı" → "car, truck, bus, motorcycle, bicycle, airplane, train, boat"
        
        YİYECEKLER:
        - "elma", "kırmızı elma", "yeşil elma" → "apple"
        - "muz", "sarı meyve" → "banana"
        - "pizza", "italyan yemeği" → "pizza"
        - "pasta", "kek", "tatlı" → "cake"
        - "donut", "halka tatlı" → "donut"
        - "sandviç", "ekmek arası" → "sandwich"
        - "portakal", "turunç" → "orange"
        - "brokoli", "yeşil sebze" → "broccoli"
        - "havuç", "turuncu sebze" → "carrot"
        - "sosisli", "hot dog" → "hot dog"
        - "yiyecek", "yemek", "besin", "gıda" → "apple, banana, pizza, cake, donut, sandwich, orange, broccoli, carrot, hot dog"
        
        EŞYALAR:
        - "sandalye", "oturak", "koltuk" → "chair"
        - "masa", "yemek masası", "çalışma masası" → "dining table"
        - "televizyon", "tv", "ekran" → "tv"
        - "laptop", "dizüstü", "bilgisayar" → "laptop"
        - "telefon", "cep telefonu", "mobil" → "cell phone"
        - "kitap", "yayın", "eser" → "book"
        - "saat", "zaman aleti" → "clock"
        - "çanta", "torba", "kese" → "handbag, backpack, suitcase"
        - "ayakkabı", "bot", "terlik" → "shoe, boot"
        - "giysi", "elbise", "kıyafet" → "clothing"
        
        ELEKTRONİK:
        - "elektronik", "teknoloji", "cihaz" → "tv, laptop, cell phone, remote, keyboard, mouse, microwave, oven, toaster, refrigerator"
        
        EV EŞYALARI:
        - "ev eşyası", "mobilya", "ev aleti" → "chair, couch, bed, dining table, toilet, tv, laptop, microwave, oven, toaster, sink, refrigerator"
        
        SADECE eşleşen COCO sınıf isimlerini virgülle ayırarak ver. Başka açıklama yapma.
        Örnek: person, car, truck
        """
        
        llm_response = self.ask_llm(llm_prompt)
        print(f"LLM sınıf eşleştirmesi: {llm_response}")
        
        lines = llm_response.strip().split('\n')
        matching_classes = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('"') and not line.startswith('Ayrıca'):
                if ',' in line:
                    classes = [cls.strip().lower() for cls in line.split(',')]
                    matching_classes.extend(classes)
                else:
                    if line.lower() in [name.lower() for name in available_classes]:
                        matching_classes.append(line.lower())
        
        if not matching_classes:
            matching_classes = [cls.strip().lower() for cls in llm_response.split(',')]
        
        print(f"Parse edilen sınıflar: {matching_classes}")
        
        # Segmentation sonuçlarını filtrele
        if hasattr(results, 'masks') and results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                class_id = int(results.boxes.cls[i])
                class_name = self.class_names[class_id]
                confidence = float(results.boxes.conf[i])
                
                if class_name.lower() in matching_classes:
                    # Mask'ı CPU'ya taşı ve numpy'a çevir
                    mask_np = mask.cpu().numpy()
                    filtered_masks.append(mask_np)
                    filtered_confidences.append(confidence)
                    filtered_classes.append(class_name)
        
        return filtered_masks, filtered_confidences, filtered_classes
    
    def filter_objects_by_color_segmentation(self, image_path, masks, confidences, classes, target_color):
        """Segmentation için renk bazında filtreleme"""
        if target_color == self.color_mapping['default']:
            return masks, confidences, classes
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return masks, confidences, classes
            
            filtered_masks = []
            filtered_confidences = []
            filtered_classes = []
            
            for i, (mask, conf, cls) in enumerate(zip(masks, confidences, classes)):
                # Mask'ı resim boyutuna uyarla
                if len(mask.shape) == 3:
                    mask = mask.squeeze()
                
                # Mask'ı 0-255 aralığına çevir
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # Mask'ı resim boyutuna yeniden boyutlandır
                mask_resized = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]))
                
                # Mask alanındaki renkleri analiz et
                mask_bool = mask_resized > 0
                if np.any(mask_bool):
                    masked_region = image[mask_bool]
                    # Ortalama renk hesapla
                    avg_color = np.mean(masked_region, axis=0)
                    
                    # Renk mesafesi hesapla
                    color_diff = np.sqrt(np.sum((avg_color - target_color) ** 2))
                    
                    # Eşik kontrolü
                    if color_diff < 200:  # Aynı eşik değeri
                        filtered_masks.append(mask)
                        filtered_confidences.append(conf)
                        filtered_classes.append(cls)
                        print(f"Segmentation renk analizi: Hedef={target_color}, Ortalama={avg_color}, Mesafe={color_diff:.1f}, Eşleşme=True")
                    else:
                        print(f"Segmentation renk analizi: Hedef={target_color}, Ortalama={avg_color}, Mesafe={color_diff:.1f}, Eşleşme=False")
                else:
                    print(f"Segmentation renk analizi: Mask boş, atlanıyor")
            
            return filtered_masks, filtered_confidences, filtered_classes
            
        except Exception as e:
            print(f"Segmentation renk filtreleme hatası: {e}")
            return masks, confidences, classes
    
    def filter_objects_by_color(self, image_path, boxes, confidences, classes, target_color):
        """Renk bazında nesne filtreleme"""
        if target_color == self.color_mapping['default']:
            return boxes, confidences, classes
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            if image is None:
                return boxes, confidences, classes
            
            filtered_boxes = []
            filtered_confidences = []
            filtered_classes = []
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = map(int, box)
                
                # Bounding box içindeki alanı al
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Renk analizi yap
                if self.is_object_color_match(roi, target_color):
                    filtered_boxes.append(box)
                    filtered_confidences.append(conf)
                    filtered_classes.append(cls)
            
            return filtered_boxes, filtered_confidences, filtered_classes
            
        except Exception as e:
            print(f"Renk filtreleme hatası: {e}")
            return boxes, confidences, classes
    
    #TODO if object coolor match ?
    def is_object_color_match(self, roi, target_color):
        """Nesnenin renginin hedef renkle eşleşip eşleşmediğini kontrol et"""
        try:
            # ROI'yi BGR'den RGB'ye çevir
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Hedef rengi BGR'den RGB'ye çevir
            target_rgb = target_color[::-1]  # BGR'den RGB'ye çevir
            
            # Renk mesafesi hesapla
            color_diff = np.sqrt(np.sum((rgb_roi - target_rgb) ** 2, axis=2))
            
            # Ortalama renk mesafesi
            avg_color_diff = np.mean(color_diff)
            
            # Renk eşleşme eşiği (0-255 arasında) - daha esnek
            color_threshold = 200
            
            # Eğer ortalama renk mesafesi eşikten küçükse eşleşme kabul et
            is_match = avg_color_diff < color_threshold
            
            print(f"Renk analizi: Hedef={target_rgb}, Ortalama mesafe={avg_color_diff:.1f}, Eşleşme={is_match}")
            
            return is_match
            
        except Exception as e:
            print(f"Renk eşleştirme hatası: {e}")
            return True  # Hata durumunda tüm nesneleri kabul et
    
    #TODO kullanici sorgusundan renk bilgisini cikarir
    def extract_color_from_query(self, user_query):
        """Kullanıcı sorgusundan renk bilgisini çıkarır"""
        user_query_lower = user_query.lower()
        
        # Renk anahtar kelimelerini ara
        for color_name, color_value in self.color_mapping.items():
            if color_name in user_query_lower and color_name != 'default':
                return color_value
        
        # Eğer hiç renk bulunamazsa varsayılan yeşil döndür
        return self.color_mapping['default']
    
    def draw_detections(self, image_path, boxes, confidences, classes, output_path, color=None):
        """Draw bounding boxes for detection mode"""
        image = cv2.imread(image_path)
        
        # Renk belirlenmemişse varsayılan yeşil kullan
        if color is None:
            color = self.color_mapping['default']
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = map(int, box)
            
            # Belirtilen renkte bounding box çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls}: {conf:.2f}"
            # Label'ı da aynı renkte yaz
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, image)
        return image
    
    #TODO drawing segmentation
    def draw_segmentation(self, image_path, masks, confidences, classes, output_path, color=None):
        """Draw segmentation masks for segmentation mode"""
        image = cv2.imread(image_path)
        
        # Renk belirlenmemişse varsayılan yeşil kullan
        if color is None:
            color = self.color_mapping['default']
        
        # Maske için şeffaflık rengi oluştur
        overlay = image.copy()
        
        for i, (mask, conf, cls) in enumerate(zip(masks, confidences, classes)):
            # Mask'ı resim boyutuna uyarla
            if len(mask.shape) == 3:
                mask = mask.squeeze()
            
            # Mask'ı 0-1 aralığında tut ve resim boyutuna uyarla
            if mask.max() > 1.0:
                mask = mask / 255.0
            
            # Mask'ı resim boyutuna yeniden boyutlandır
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            
            # Renkli maske oluştur
            colored_mask = np.zeros_like(image)
            colored_mask[mask_uint8 > 0] = color
            
            # Maske üzerine çiz
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
            # Bounding box hesapla
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                
                # Bounding box çiz
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Label ekle
                label = f"{cls}: {conf:.2f}"
                cv2.putText(overlay, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, overlay)
        return overlay
    
    #TODO interact with llm model
    def ask_llm(self, prompt):
        try:
            response = ollama.chat(model='llama3.1:latest', messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            return f"LLM hatası: {str(e)}"
    
    #TODO finalize every part in here
    def process_image(self, image_path, user_query):
        print(f"Görüntü işleniyor: {image_path}")
        print(f"Kullanıcı sorgusu: {user_query}")
        print(f"Mod: {self.mode}")
        
        # Kullanıcı sorgusundan renk bilgisini çıkar
        detected_color = self.extract_color_from_query(user_query)
        color_name = [name for name, value in self.color_mapping.items() if value == detected_color and name != 'default'][0]
        print(f"Tespit edilen renk: {color_name}")
        
        results = self.detect_objects(image_path)
        
        if self.mode == 'segmentation':
            # Segmentation modu
            masks, confidences, classes = self.filter_objects_by_class_segmentation(results, user_query)
            
            # Renk bazında filtrele (eğer renk belirtilmişse)
            if detected_color != self.color_mapping['default']:
                print(f"Renk filtreleme uygulanıyor: {color_name}")
                masks, confidences, classes = self.filter_objects_by_color_segmentation(image_path, masks, confidences, classes, detected_color)
            
            output_path = "output_segmentation.jpg"
            if masks:
                self.draw_segmentation(image_path, masks, confidences, classes, output_path, detected_color)
                print(f"Tespit edilen nesneler: {classes}")
                print(f"Sonuç görüntüsü kaydedildi: {output_path}")
                print(f"Segmentation rengi: {color_name}")
            else:
                print("Belirtilen nesneler bulunamadı.")
            
            return masks, confidences, classes
        else:
            # Detection modu
            boxes, confidences, classes = self.filter_objects_by_class(results, user_query)
            
            # Renk bazında filtrele (eğer renk belirtilmişse)
            if detected_color != self.color_mapping['default']:
                print(f"Renk filtreleme uygulanıyor: {color_name}")
                boxes, confidences, classes = self.filter_objects_by_color(image_path, boxes, confidences, classes, detected_color)
            
            output_path = "output_detection.jpg"
            if boxes:
                self.draw_detections(image_path, boxes, confidences, classes, output_path, detected_color)
                print(f"Tespit edilen nesneler: {classes}")
                print(f"Sonuç görüntüsü kaydedildi: {output_path}")
                print(f"Bounding box rengi: {color_name}")
            else:
                print("Belirtilen nesneler bulunamadı.")
            
            return boxes, confidences, classes

class VideoProcessor:
    def __init__(self, detector):
        """
        Initialize Video Processor
        Args:
            detector: VLMDetector instance
        """
        self.detector = detector
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    def is_video_file(self, file_path):
        """Check if file is a supported video format"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def get_video_info(self, video_path):
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info
    
    def process_video_frames(self, video_path, user_query, output_dir="video_output", 
                           frame_skip=1, max_frames=None):
        """
        Process video frames for detection
        Args:
            video_path: Path to video file
            user_query: Turkish query for detection
            output_dir: Directory to save results
            frame_skip: Process every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to process
        """
        print(f"Video işleniyor: {video_path}")
        print(f"Kullanıcı sorgusu: {user_query}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video info
        video_info = self.get_video_info(video_path)
        if not video_info:
            print("Video dosyası açılamadı!")
            return None
        
        print(f"Video bilgileri: {video_info['width']}x{video_info['height']}, "
              f"{video_info['fps']:.2f} FPS, {video_info['duration']:.2f} saniye")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Video açılamadı!")
            return None
        
        # Setup video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, f"detected_{Path(video_path).stem}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, video_info['fps'], 
                            (video_info['width'], video_info['height']))
        
        frame_count = 0
        processed_frames = 0
        detection_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Limit max frames
                if max_frames and processed_frames >= max_frames:
                    break
                
                print(f"Frame {frame_count + 1}/{video_info['frame_count']} işleniyor...")
                
                # Save frame temporarily
                temp_frame_path = os.path.join(output_dir, f"temp_frame_{frame_count}.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                # Process frame
                if self.detector.mode == 'segmentation':
                    masks, confidences, classes = self.detector.process_image(temp_frame_path, user_query)
                    if masks:
                        # Draw segmentation on frame
                        annotated_frame = self.detector.draw_segmentation(
                            temp_frame_path, masks, confidences, classes, 
                            temp_frame_path, self.detector.extract_color_from_query(user_query)
                        )
                        annotated_frame = cv2.imread(temp_frame_path)
                    else:
                        annotated_frame = frame
                else:
                    boxes, confidences, classes = self.detector.process_image(temp_frame_path, user_query)
                    if boxes:
                        # Draw detections on frame
                        annotated_frame = self.detector.draw_detections(
                            temp_frame_path, boxes, confidences, classes, 
                            temp_frame_path, self.detector.extract_color_from_query(user_query)
                        )
                        annotated_frame = cv2.imread(temp_frame_path)
                    else:
                        annotated_frame = frame
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Store results
                detection_results.append({
                    'frame': frame_count,
                    'objects': classes if 'classes' in locals() else [],
                    'count': len(classes) if 'classes' in locals() else 0
                })
                
                # Clean up temp file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                
                processed_frames += 1
                frame_count += 1
        
        except KeyboardInterrupt:
            print("Video işleme durduruldu!")
        
        finally:
            cap.release()
            out.release()
        
        # Save detection summary
        summary_path = os.path.join(output_dir, f"detection_summary_{Path(video_path).stem}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'video_info': video_info,
                'query': user_query,
                'processed_frames': processed_frames,
                'total_frames': video_info['frame_count'],
                'frame_skip': frame_skip,
                'detection_results': detection_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Video işleme tamamlandı!")
        print(f"İşlenen frame sayısı: {processed_frames}")
        print(f"Çıktı video: {output_path}")
        print(f"Özet dosyası: {summary_path}")
        
        return {
            'output_video': output_path,
            'summary': summary_path,
            'processed_frames': processed_frames,
            'total_frames': video_info['frame_count']
        }
    
    def process_webcam(self, user_query, duration=30, output_path="webcam_output.mp4"):
        """
        Process webcam feed for real-time detection
        Args:
            user_query: Turkish query for detection
            duration: Duration in seconds (0 = infinite)
            output_path: Output video path
        """
        print(f"Webcam başlatılıyor...")
        print(f"Kullanıcı sorgusu: {user_query}")
        print(f"Süre: {duration} saniye" if duration > 0 else "Süre: Sınırsız")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam açılamadı!")
            return None
        
        # Get webcam properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Webcam çözünürlük: {width}x{height}, FPS: {fps}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check duration limit
                if duration > 0 and (time.time() - start_time) > duration:
                    break
                
                # Process every 5th frame for performance
                if frame_count % 5 == 0:
                    # Save frame temporarily
                    temp_frame_path = "temp_webcam_frame.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Process frame
                    if self.detector.mode == 'segmentation':
                        masks, confidences, classes = self.detector.process_image(temp_frame_path, user_query)
                        if masks:
                            annotated_frame = self.detector.draw_segmentation(
                                temp_frame_path, masks, confidences, classes, 
                                temp_frame_path, self.detector.extract_color_from_query(user_query)
                            )
                            annotated_frame = cv2.imread(temp_frame_path)
                        else:
                            annotated_frame = frame
                    else:
                        boxes, confidences, classes = self.detector.process_image(temp_frame_path, user_query)
                        if boxes:
                            annotated_frame = self.detector.draw_detections(
                                temp_frame_path, boxes, confidences, classes, 
                                temp_frame_path, self.detector.extract_color_from_query(user_query)
                            )
                            annotated_frame = cv2.imread(temp_frame_path)
                        else:
                            annotated_frame = frame
                    
                    # Clean up temp file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                else:
                    annotated_frame = frame
                
                # Add frame info
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Time: {int(time.time() - start_time)}s", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame
                out.write(annotated_frame)
                
                # Show frame (optional)
                cv2.imshow('Webcam Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("Webcam işleme durduruldu!")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Webcam işleme tamamlandı!")
        print(f"Toplam frame: {frame_count}")
        print(f"Çıktı video: {output_path}")
        
        return output_path

#TODO main function
def main():
    detector = VLMDetector()
    video_processor = VideoProcessor(detector)
    
    print("🎯 English-Turkish VLM Detector")
    print("1. Resim işleme")
    print("2. Video işleme")
    print("3. Webcam işleme")
    
    choice = input("Seçiminizi yapın (1-3): ").strip()
    
    if choice == "1":
        # Image processing
        image_path = input("Görüntü dosyasının yolunu girin: ")
        user_query = input("Ne aramak istiyorsunuz? (örn: 'mavi arabaları göster', 'kırmızı kedileri bul'): ")
        detector.process_image(image_path, user_query)
    
    elif choice == "2":
        # Video processing
        video_path = input("Video dosyasının yolunu girin: ")
        if not video_processor.is_video_file(video_path):
            print("Desteklenmeyen video formatı!")
            return
        
        user_query = input("Ne aramak istiyorsunuz? (örn: 'mavi arabaları göster', 'kırmızı kedileri bul'): ")
        
        # Video processing options
        frame_skip = input("Frame atlama (1 = tüm frameler, 5 = her 5. frame): ").strip()
        frame_skip = int(frame_skip) if frame_skip.isdigit() else 1
        
        max_frames = input("Maksimum frame sayısı (boş = sınırsız): ").strip()
        max_frames = int(max_frames) if max_frames.isdigit() else None
        
        video_processor.process_video_frames(video_path, user_query, frame_skip=frame_skip, max_frames=max_frames)
    
    elif choice == "3":
        # Webcam processing
        user_query = input("Ne aramak istiyorsunuz? (örn: 'mavi arabaları göster', 'kırmızı kedileri bul'): ")
        
        duration = input("Kayıt süresi (saniye, 0 = sınırsız): ").strip()
        duration = int(duration) if duration.isdigit() else 30
        
        video_processor.process_webcam(user_query, duration=duration)
    
    else:
        print("Geçersiz seçim!")

if __name__ == "__main__":
    main()
