import ExpoModulesCore
import Vision
import CoreML
import UIKit
import Accelerate

// =============================================================================
// Python run_v3_all.py mantığının birebir Swift karşılığı.
//
// Python'da olan:
//   results = model.predict(img, conf=0.25, imgsz=1024, retina_masks=True)
//   result.masks.xy  →  polygon noktaları
//   cv2.fillPoly + cv2.drawContours + cv2.addWeighted
//
// Swift'te birebir aynısı:
//   1. Letterbox (CIContext, thread-safe)
//   2. MLModel.prediction (doğrudan CoreML, Vision yok)
//   3. Decode YOLO (sigmoid + NMS)
//   4. Mask prototype × coefficients → binary mask → polygon
//   5. UIBezierPath fill + stroke
// =============================================================================

public class EyeglassDetectorModule: Module {
  
  private static var cachedModel: MLModel?
  private static var modelLoadError: String?

  public func definition() -> ModuleDefinition {
    Name("EyeglassDetector")

    AsyncFunction("analyzeImage") { (imageUri: String, promise: Promise) in
      DispatchQueue.global(qos: .userInitiated).async {
        do {
          // ── 1. Görüntüyü yükle (Python: img = cv2.imread(path)) ──
          guard let url = URL(string: imageUri),
                let imageData = try? Data(contentsOf: url),
                let uiImage = UIImage(data: imageData) else {
            promise.reject("ERR_IMAGE", "Cannot load image from URI")
            return
          }

          // ── 2. Model yükle ──
          let mlModel = try self.getModel()
          
          // Model'in beklediği boyutu oku (Python: imgsz=1024)
          var targetSize = CGSize(width: 1024, height: 1024)
          if let inputDesc = mlModel.modelDescription.inputDescriptionsByName["image"],
             let ic = inputDesc.imageConstraint {
            targetSize = CGSize(width: CGFloat(ic.pixelsWide), height: CGFloat(ic.pixelsHigh))
          }
          NSLog("[EyeglassDetector] Model targetSize=%.0fx%.0f", targetSize.width, targetSize.height)
          
          // ── 3. Letterbox (Python: Ultralytics iç letterbox fonksiyonu) ──
          let (bufferOpt, scale, padX, padY, normalizedImage) = self.createLetterboxedPixelBuffer(from: uiImage, targetSize: targetSize)
          guard let buffer = bufferOpt else {
            promise.reject("ERR", "Failed to create CVPixelBuffer")
            return
          }
          
          // ── 4. Inference (Python: model.predict()) ──
          let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
          let prediction = try mlModel.prediction(from: input)
          
          // ── 5. Çıktı tensörlerini bul ──
          var rawOutput: MLMultiArray?
          var protoOutput: MLMultiArray?
          
          for featureName in prediction.featureNames {
            if let ma = prediction.featureValue(for: featureName)?.multiArrayValue {
              if ma.shape.count == 3 && ma.shape[2].intValue > 1000 {
                rawOutput = ma  // [1, 4+nc+32, 8400]
              } else if ma.shape.count == 4 {
                protoOutput = ma  // [1, 32, maskH, maskW]
              }
            }
          }
          
          guard let output = rawOutput else {
            promise.reject("ERR", "Model output not found")
            return
          }
          
          NSLog("[EyeglassDetector] output=%@, protos=%@", "\(output.shape)", "\(protoOutput?.shape ?? [])")
          
          // ── 6. Decode YOLO (Python: Ultralytics postprocess) ──
          let detections = self.decodeYOLO(output: output, scale: scale, padX: padX, padY: padY, origSize: uiImage.size)
          NSLog("[EyeglassDetector] %d detections after NMS", detections.count)
          
          // ── 7. Maske çiz (Python: cv2.fillPoly + drawContours + addWeighted) ──
          let finalImage = self.drawResults(
            image: normalizedImage,
            detections: detections,
            prototypes: protoOutput,
            scale: scale, padX: padX, padY: padY,
            modelSize: targetSize
          )
          
          if let jpeg = finalImage.jpegData(compressionQuality: 0.9) {
            let b64 = jpeg.base64EncodedString()
            promise.resolve("data:image/jpeg;base64,\(b64)")
            return
          }
          
          promise.reject("ERR", "Failed to encode result image")
        } catch {
          promise.reject("ERR_INFERENCE", error.localizedDescription)
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Letterbox (Python: Ultralytics LetterBox transform)
  // CGContext + explicit sRGB — CIContext renk uzayı dönüşüm sorununu çözer
  // ═══════════════════════════════════════════════════════════════════════════
  private func createLetterboxedPixelBuffer(from image: UIImage, targetSize: CGSize) -> (CVPixelBuffer?, CGFloat, CGFloat, CGFloat, UIImage) {
    let origSize = image.size
    let imgScale = min(targetSize.width / origSize.width, targetSize.height / origSize.height)
    let newW = origSize.width * imgScale
    let newH = origSize.height * imgScale
    let padX = (targetSize.width - newW) / 2.0
    let padY = (targetSize.height - newH) / 2.0
    
    // ── Step 1: UIImage orientation'ını normalize et → doğru CGImage elde et ──
    UIGraphicsBeginImageContextWithOptions(image.size, true, 1.0)
    if let ctx = UIGraphicsGetCurrentContext() {
      // image.draw(at:) will automatically respect the EXIF orientation metadata.
      // We don't need to manually flip it, otherwise it causes a double-mirror effect.
    }
    image.draw(at: .zero)
    guard let normalizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
      UIGraphicsEndImageContext()
      return (nil, imgScale, padX, padY, image)
    }
    UIGraphicsEndImageContext()
    
    guard let cgImage = normalizedImage.cgImage else {
      NSLog("[EyeglassDetector] ERROR: Failed to normalize image orientation")
      return (nil, imgScale, padX, padY, normalizedImage)
    }
    
    // ── Step 2: CVPixelBuffer oluştur (32BGRA) ──
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(targetSize.width), Int(targetSize.height),
                                     kCVPixelFormatType_32BGRA, attrs, &pb)
    guard status == kCVReturnSuccess, let buffer = pb else { return (nil, imgScale, padX, padY, normalizedImage) }
    
    // ── Step 3: CGContext ile doğrudan çiz (sRGB, renk dönüşümü yok!) ──
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    
    guard let pixelData = CVPixelBufferGetBaseAddress(buffer) else { return (nil, imgScale, padX, padY, normalizedImage) }
    
    let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    // BGRA = byteOrder32Little + premultipliedFirst
    let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    guard let context = CGContext(data: pixelData,
                                  width: Int(targetSize.width),
                                  height: Int(targetSize.height),
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: srgb,
                                  bitmapInfo: bitmapInfo)
    else { return (nil, imgScale, padX, padY, normalizedImage) }
    
    // Padding rengi (Ultralytics: RGB 114,114,114)
    context.setFillColor(red: 114.0/255.0, green: 114.0/255.0, blue: 114.0/255.0, alpha: 1.0)
    context.fill(CGRect(origin: .zero, size: targetSize))
    
    // CGContext varsayılan Core Graphics (Y-UP) koordinat sistemindedir.
    // CGContext.draw() bu sistemde görüntüyü belleğe düz (right-side up) olarak yazar.
    context.draw(cgImage, in: CGRect(x: padX, y: padY, width: newW, height: newH))
    
    // ── DEBUG: Piksel değerlerini logla (model ne görüyor?) ──
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let raw = pixelData.assumingMemoryBound(to: UInt8.self)
    // Ortadaki piksel
    let centerX = Int(targetSize.width) / 2
    let centerY = Int(targetSize.height) / 2
    let offset = centerY * bytesPerRow + centerX * 4
    let b = raw[offset], g = raw[offset+1], r = raw[offset+2], a = raw[offset+3]
    NSLog("[EyeglassDetector] Pixel at center (%d,%d): B=%d G=%d R=%d A=%d", centerX, centerY, b, g, r, a)
    // Padding pikseli (sol üst köşe)
    let p0 = raw[0], p1 = raw[1], p2 = raw[2], p3 = raw[3]
    NSLog("[EyeglassDetector] Pixel at (0,0) padding: B=%d G=%d R=%d A=%d", p0, p1, p2, p3)
    
    NSLog("[EyeglassDetector] Letterbox: %.0fx%.0f → %.0fx%.0f scale=%.4f pad=(%.1f,%.1f)",
          origSize.width, origSize.height, targetSize.width, targetSize.height, imgScale, padX, padY)
    return (buffer, imgScale, padX, padY, normalizedImage)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Decode YOLO (Python: Ultralytics non_max_suppression)
  // Girdi: [1, 4+nc+32, 8400] → Çıktı: NMS uygulanmış Detection listesi
  // ═══════════════════════════════════════════════════════════════════════════
  private struct Detection {
    let rect: CGRect        // Orijinal görüntü koordinatlarında bbox
    let confidence: Float
    let classIdx: Int
    let label: String
    let maskCoeffs: [Float] // 32 mask katsayısı
    let rawCx: Float, rawCy: Float, rawW: Float, rawH: Float // Model koordinatlarında (unpad öncesi)
  }
  
  private func decodeYOLO(output: MLMultiArray, scale: CGFloat, padX: CGFloat, padY: CGFloat, origSize: CGSize) -> [Detection] {
    let shape = output.shape.map { $0.intValue }
    let strides = output.strides.map { $0.intValue }
    
    let numFeatures = shape[1]  // 4 + nc + 32
    let numAnchors = shape[2]   // 8400
    let numClasses = numFeatures - 36  // nc = numFeatures - 4(bbox) - 32(mask)
    let maskStart = 4 + numClasses
    
    var isFloat16 = false
    if #available(iOS 16.0, *) {
      isFloat16 = output.dataType == .float16
    } else {
      isFloat16 = output.dataType.rawValue == 65552
    }
    
    let ptr16 = isFloat16 ? output.dataPointer.bindMemory(to: UInt16.self, capacity: output.count) : nil
    let ptr32 = isFloat16 ? nil : output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
    
    // Helper to safely read float values from either Float32 or Float16 buffer
    func readVal(offset: Int) -> Float {
      if let p32 = ptr32 { return p32[offset] }
      if let p16 = ptr16 {
        let h = p16[offset]
        var t1 = UInt32(h & 0x7fff) << 13
        let e = h & 0x7c00
        if e == 0 {
            if t1 != 0 { /* subnormal handling omitted for speed, it's ~0 anyway */ }
        } else if e == 0x7c00 {
            t1 |= 0x7f800000
        } else {
            t1 += 0x38000000
        }
        t1 |= UInt32(h & 0x8000) << 16
        return Float(bitPattern: t1)
      }
      return 0
    }
    
    let s1 = strides[1]  // stride along feature dim
    let s2 = strides[2]  // stride along anchor dim
    
    // Python: conf=0.25
    let confThreshold: Float = 0.25
    var detections: [Detection] = []
    
    let classNames = numClasses > 1 ? ["frame", "glass"] : ["glass"]
    
    for a in 0..<numAnchors {
      // YOLOv8 sınıfları birbirinden bağımsızdır (mutually exclusive değildir).
      // Masaüstündeki gibi sadece 'glass' (class 1) sınıfı üzerinde filtreleme yapıyoruz.
      let classIdx = 1
      guard classIdx < numClasses else { continue }
      
      let conf = readVal(offset: (4 + classIdx) * s1 + a * s2)
      if conf < confThreshold { continue }
      
      let maxConf = conf
      let bestClass = classIdx
      
      // Bounding box (model koordinatları, pad dahil)
      let cx = readVal(offset: 0 * s1 + a * s2)
      let cy = readVal(offset: 1 * s1 + a * s2)
      let w  = readVal(offset: 2 * s1 + a * s2)
      let h  = readVal(offset: 3 * s1 + a * s2)
      
      // Orijinal görüntü koordinatlarına çevir (unpad + unscale)
      let ox = (CGFloat(cx) - padX) / scale
      let oy = (CGFloat(cy) - padY) / scale
      let ow = CGFloat(w) / scale
      let oh = CGFloat(h) / scale
      let rect = CGRect(x: ox - ow/2, y: oy - oh/2, width: ow, height: oh)
      
      if rect.width < 5 || rect.height < 5 { continue }
      
      // Mask katsayıları (32 adet)
      var coeffs = [Float](repeating: 0, count: 32)
      for m in 0..<32 { coeffs[m] = readVal(offset: (maskStart + m) * s1 + a * s2) }
      
      // Yalnızca 'glass' sınıfını (class 1) tut (çerçeveyi gösterme)
      if bestClass != 1 { continue }
      
      let label = classNames.count > 1 ? classNames[bestClass] : "unknown"
      detections.append(Detection(rect: rect, confidence: maxConf, classIdx: bestClass,
                                  label: label, maskCoeffs: coeffs,
                                  rawCx: cx, rawCy: cy, rawW: w, rawH: h))
    }
    
    // İlk 5 detection logla
    for (i, d) in detections.prefix(5).enumerated() {
      NSLog("[EyeglassDetector] det#%d: conf=%.3f %@ cx=%.1f cy=%.1f w=%.1f h=%.1f",
            i, d.confidence, d.label, d.rawCx, d.rawCy, d.rawW, d.rawH)
    }
    NSLog("[EyeglassDetector] %d detections before NMS", detections.count)
    
    return applyNMS(detections: detections, iouThreshold: 0.45)
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Maske Çiz (Python: cv2.fillPoly + drawContours + addWeighted)
  // ═══════════════════════════════════════════════════════════════════════════
  private func drawResults(image: UIImage, detections: [Detection], prototypes: MLMultiArray?,
                           scale: CGFloat, padX: CGFloat, padY: CGFloat, modelSize: CGSize) -> UIImage {
    if detections.isEmpty {
      NSLog("[EyeglassDetector] No detections, returning original")
      return image
    }
    
    let imgW = image.size.width
    let imgH = image.size.height
    
    // Python: out = img.copy() + overlay = img.copy()
    UIGraphicsBeginImageContextWithOptions(image.size, true, 1.0)
    image.draw(at: .zero)
    guard let ctx = UIGraphicsGetCurrentContext() else {
      UIGraphicsEndImageContext()
      return image
    }
    
    // Çizim kalitesini en üst düzeye çıkar
    ctx.setAllowsAntialiasing(true)
    ctx.setShouldAntialias(true)
    ctx.interpolationQuality = .high
    
    for det in detections {
      let color: UIColor = det.label == "frame"
        ? UIColor(red: 50/255.0, green: 205/255.0, blue: 50/255.0, alpha: 1.0)
        : UIColor(red: 30/255.0, green: 144/255.0, blue: 255/255.0, alpha: 1.0)
      
      // ── Maske ──
      if let protos = prototypes {
        let maskImage = generateMask(detection: det, prototypes: protos,
                                     scale: scale, padX: padX, padY: padY,
                                     origW: Int(imgW), origH: Int(imgH),
                                     modelSize: modelSize)
        if let mask = maskImage {
          // Python: cv2.fillPoly(overlay, [pts], color) + addWeighted(overlay, 0.4, ...)
          // Python: cv2.drawContours(out, [pts], -1, color, 2)
          if let contourPath = extractContourPath(from: mask, imageSize: image.size) {
            let path = UIBezierPath(cgPath: contourPath)
            path.lineJoinStyle = .round
            path.lineCapStyle = .round
            
            // Sadece sınırları (kenarları) çiz, içini doldurma
            color.setStroke()
            path.lineWidth = 3.0 // Kenarları biraz daha belirginleştirdik
            path.stroke()
          }
        }
      }
    }
    
    let result = UIGraphicsGetImageFromCurrentImageContext() ?? image
    UIGraphicsEndImageContext()
    return result
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Mask Generation (Python: result.masks.data → retina_masks)
  //
  // Python mantığı:
  //   mask = (mask_coeffs @ prototypes).reshape(maskH, maskW)  # matris çarpımı
  //   mask = sigmoid(mask)
  //   mask = crop_mask(mask, bbox)
  //   mask = F.interpolate(mask, origSize)   # retina_masks=True
  //   mask = mask > 0.5
  // ═══════════════════════════════════════════════════════════════════════════
  private func generateMask(detection: Detection, prototypes: MLMultiArray,
                            scale: CGFloat, padX: CGFloat, padY: CGFloat,
                            origW: Int, origH: Int, modelSize: CGSize) -> CGImage? {
    let shape = prototypes.shape.map { $0.intValue }
    let strides = prototypes.strides.map { $0.intValue }
    let numProtos = shape[1]  // 32
    let maskH = shape[2]      // e.g. 160 or 256
    let maskW = shape[3]      // e.g. 160 or 256
    let protoStride = strides[1]  // stride for proto channel dim
    
    var isFloat16 = false
    if #available(iOS 16.0, *) {
      isFloat16 = prototypes.dataType == .float16
    } else {
      isFloat16 = prototypes.dataType.rawValue == 65552
    }
    
    let pPtr16 = isFloat16 ? prototypes.dataPointer.bindMemory(to: UInt16.self, capacity: prototypes.count) : nil
    let pPtr32 = isFloat16 ? nil : prototypes.dataPointer.bindMemory(to: Float.self, capacity: prototypes.count)
    
    func readProto(offset: Int) -> Float {
      if let p32 = pPtr32 { return p32[offset] }
      if let p16 = pPtr16 {
        let h = p16[offset]
        var t1 = UInt32(h & 0x7fff) << 13
        let e = h & 0x7c00
        if e == 0 {
            if t1 != 0 { /* subnormal handling omitted */ }
        } else if e == 0x7c00 {
            t1 |= 0x7f800000
        } else {
            t1 += 0x38000000
        }
        t1 |= UInt32(h & 0x8000) << 16
        return Float(bitPattern: t1)
      }
      return 0
    }
    
    // ── Step 1: mask = coeffs @ prototypes (Python: mask_coeffs @ prototypes) ──
    // Sonuç: maskH x maskW boyutunda bir matris
    var maskData = [Float](repeating: 0, count: maskH * maskW)
    for c in 0..<min(numProtos, 32) {
      let coeff = detection.maskCoeffs[c]
      if abs(coeff) < 1e-7 { continue }  // sıfır katsayıları atla (performans)
      let offset = c * protoStride
      for i in 0..<(maskH * maskW) {
        maskData[i] += coeff * readProto(offset: offset + i)
      }
    }
    
    // ── Step 2: sigmoid (Python: sigmoid otomatik) ──
    for i in 0..<maskData.count {
      maskData[i] = sigmoid(maskData[i])
    }
    
    // ── Step 3: crop_mask — Bounding box dışını sıfırla ──
    // Model koordinatlarındaki bbox'ı mask grid koordinatlarına çevir
    let scaleToMaskX = Float(maskW) / Float(modelSize.width)
    let scaleToMaskY = Float(maskH) / Float(modelSize.height)
    
    let bboxX1 = (detection.rawCx - detection.rawW / 2) * scaleToMaskX
    let bboxY1 = (detection.rawCy - detection.rawH / 2) * scaleToMaskY
    let bboxX2 = (detection.rawCx + detection.rawW / 2) * scaleToMaskX
    let bboxY2 = (detection.rawCy + detection.rawH / 2) * scaleToMaskY
    
    for y in 0..<maskH {
      for x in 0..<maskW {
        if Float(x) < bboxX1 || Float(x) > bboxX2 || Float(y) < bboxY1 || Float(y) > bboxY2 {
          maskData[y * maskW + x] = 0
        }
      }
    }
    
    // ── Step 4: Retina mask — Orijinal çözünürlüğe ölçekle ──
    // Python: F.interpolate(mask, (origH, origW), mode='bilinear')
    // Ancak önce unpad yapmalıyız (letterbox padding'i çıkar)
    // mask grid → model pixel → unpad → orig pixel
    
    // Orijinal boyutta binary maske oluştur
    var origMask = [UInt8](repeating: 0, count: origW * origH)
    
    for oy in 0..<origH {
      for ox in 0..<origW {
        // Orijinal piksel → model piksel (letterbox koordinatları)
        let mx = Float(ox) * Float(scale) + Float(padX)
        let my = Float(oy) * Float(scale) + Float(padY)
        
        // Model piksel → mask grid piksel
        let fx = mx * scaleToMaskX
        let fy = my * scaleToMaskY
        
        let x1 = Int(fx)
        let y1 = Int(fy)
        let x2 = min(x1 + 1, maskW - 1)
        let y2 = min(y1 + 1, maskH - 1)
        
        // Maske gridi sınırları içindeyse bilinear interpolation yap
        if x1 >= 0 && x1 < maskW && y1 >= 0 && y1 < maskH {
          let dx = fx - Float(x1)
          let dy = fy - Float(y1)
          
          let v11 = maskData[y1 * maskW + x1]
          let v12 = maskData[y1 * maskW + x2]
          let v21 = maskData[y2 * maskW + x1]
          let v22 = maskData[y2 * maskW + x2]
          
          let top = v11 + (v12 - v11) * dx
          let bottom = v21 + (v22 - v21) * dx
          let val = top + (bottom - top) * dy
          
          // Python: mask > 0.5
          if val > 0.5 {
            origMask[oy * origW + ox] = 255
          }
        }
      }
    }
    
    // CGImage oluştur (grayscale mask)
    let maskDataObj = Data(origMask)
    let colorSpace = CGColorSpaceCreateDeviceGray()
    guard let provider = CGDataProvider(data: maskDataObj as CFData),
          let cgMask = CGImage(width: origW, height: origH,
                               bitsPerComponent: 8, bitsPerPixel: 8,
                               bytesPerRow: origW, space: colorSpace,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                               provider: provider, decode: nil,
                               shouldInterpolate: false, intent: .defaultIntent)
    else { return nil }
    
    return cgMask
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Kontur Çıkarma (Python: cv2.drawContours)
  // Binary mask'tan sınır çizgisi çıkar
  // ═══════════════════════════════════════════════════════════════════════════
  private func extractContourPath(from mask: CGImage, imageSize: CGSize) -> CGPath? {
    if #available(iOS 14.0, *) {
      let request = VNDetectContoursRequest()
      request.contrastAdjustment = 1.0
      request.detectsDarkOnLight = false
      
      let handler = VNImageRequestHandler(cgImage: mask, options: [:])
      do {
        try handler.perform([request])
        if let obs = request.results?.first as? VNContoursObservation {
          let path = CGMutablePath()
          for i in 0..<obs.contourCount {
            let contour = try obs.contour(at: i)
            path.addPath(contour.normalizedPath)
          }
          if path.boundingBox.width < 0.001 { return nil }
          
          // Vision normalized [0,1] → orijinal piksel koordinatları
          // Vision Y ekseni: 0 = alt, 1 = üst → UIKit: 0 = üst
          var t = CGAffineTransform(scaleX: imageSize.width, y: -imageSize.height)
          t = t.translatedBy(x: 0, y: -1.0)
          return path.copy(using: &t)
        }
      } catch { }
    }
    return nil
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - NMS (Python: Ultralytics non_max_suppression)
  // ═══════════════════════════════════════════════════════════════════════════
  private func applyNMS(detections: [Detection], iouThreshold: Float) -> [Detection] {
    let sorted = detections.sorted { $0.confidence > $1.confidence }
    var kept: [Detection] = []
    var suppressed = Set<Int>()
    for i in 0..<sorted.count {
      if suppressed.contains(i) { continue }
      kept.append(sorted[i])
      for j in (i+1)..<sorted.count {
        if suppressed.contains(j) { continue }
        if computeIoU(sorted[i].rect, sorted[j].rect) > CGFloat(iouThreshold) {
          suppressed.insert(j)
        }
      }
    }
    return kept
  }
  
  private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
    let inter = a.intersection(b)
    if inter.isNull { return 0 }
    let iA = inter.width * inter.height
    let uA = a.width * a.height + b.width * b.height - iA
    return uA > 0 ? iA / uA : 0
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Yardımcılar
  // ═══════════════════════════════════════════════════════════════════════════
  private func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
  }
  
  private func imageOrientationToCGOrientation(_ o: UIImage.Orientation) -> CGImagePropertyOrientation {
    switch o {
    case .up: return .up
    case .upMirrored: return .upMirrored
    case .down: return .down
    case .downMirrored: return .downMirrored
    case .left: return .left
    case .leftMirrored: return .leftMirrored
    case .right: return .right
    case .rightMirrored: return .rightMirrored
    @unknown default: return .up
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // MARK: - Model Yükleme
  // ÖNCELİK: .mlpackage (eğitilmiş) > .mlmodelc (eski cache)
  // ═══════════════════════════════════════════════════════════════════════════
  private func getModel() throws -> MLModel {
    if let cached = EyeglassDetectorModule.cachedModel { return cached }
    if let err = EyeglassDetectorModule.modelLoadError {
      throw NSError(domain: "EyeglassDetector", code: -1, userInfo: [NSLocalizedDescriptionKey: err])
    }
    
    let bundles: [Bundle] = [
      Bundle(for: EyeglassDetectorModule.self), Bundle.main,
      Bundle(for: EyeglassDetectorModule.self)
        .url(forResource: "EyeglassDetector", withExtension: "bundle")
        .flatMap { Bundle(url: $0) }
    ].compactMap { $0 }
    
    var modelUrl: URL?
    for b in bundles {
      // Önce .mlpackage (eğitilmiş model)
      if let u = b.url(forResource: "best", withExtension: "mlpackage") {
        NSLog("[EyeglassDetector] Found mlpackage: %@", u.lastPathComponent)
        modelUrl = try MLModel.compileModel(at: u)
        break
      }
      // Fallback: .mlmodelc
      if let u = b.url(forResource: "best", withExtension: "mlmodelc") {
        NSLog("[EyeglassDetector] Found mlmodelc: %@", u.lastPathComponent)
        modelUrl = u
        break
      }
    }
    
    guard let url = modelUrl else {
      let m = "CoreML model not found in any bundle"
      EyeglassDetectorModule.modelLoadError = m
      throw NSError(domain: "EyeglassDetector", code: -2, userInfo: [NSLocalizedDescriptionKey: m])
    }
    
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndGPU
    let mlModel = try MLModel(contentsOf: url, configuration: config)
    
    if let inputDesc = mlModel.modelDescription.inputDescriptionsByName["image"],
       let ic = inputDesc.imageConstraint {
      NSLog("[EyeglassDetector] Model loaded: %dx%d", ic.pixelsWide, ic.pixelsHigh)
    }
    
    EyeglassDetectorModule.cachedModel = mlModel
    return mlModel
  }
}
