import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  FlatList,
  Dimensions,
  StatusBar,
  Alert,
  ActivityIndicator,
  Animated,
  Platform,
  Modal,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { BlurView } from 'expo-blur';
import { Ionicons } from '@expo/vector-icons';

const { width: W, height: H } = Dimensions.get('window');
const API_URL = 'http://127.0.0.1:5000';

const C = {
  bg: '#000000',
  card: '#1C1C1E', // iOS Dark Mode Card
  blue: '#0A84FF', // iOS Dark Mode Blue
  white: '#FFFFFF',
  textSecondary: '#EBEBF599', // iOS Secondary Label (Dark)
  border: '#38383A', // iOS Separator (Dark)
};

type PhotoItem = {
  id: string;
  uri: string;
  timestamp: number;
  analyzed: boolean;
  resultUri?: string;
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photos, setPhotos] = useState<PhotoItem[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<PhotoItem | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showCamera, setShowCamera] = useState(true);
  const cameraRef = useRef<CameraView>(null);
  
  const [showGallery, setShowGallery] = useState(false);
  const [galleryImages, setGalleryImages] = useState<any[]>([]);
  const [processedImages, setProcessedImages] = useState<any[]>([]);
  const [galleryTab, setGalleryTab] = useState<'raw'|'processed'>('raw');

  // Animations
  const shutterAnim = useRef(new Animated.Value(1)).current;

  const handleShutterPressIn = () => {
    Animated.spring(shutterAnim, { toValue: 0.9, useNativeDriver: true }).start();
  };
  const handleShutterPressOut = () => {
    Animated.spring(shutterAnim, { toValue: 1, friction: 3, useNativeDriver: true }).start();
  };

  const takePhoto = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      if (photo) {
        Alert.alert("Camera", "Local capture requires full YOLO integration. Use Gallery to test API.");
      }
    } catch (e) {
      Alert.alert('Error', 'Failed to capture image.');
    }
  };

  const fetchGallery = async () => {
    try {
      const [resRaw, resProc] = await Promise.all([
        fetch(`${API_URL}/gallery`),
        fetch(`${API_URL}/gallery/processed`)
      ]);
      setGalleryImages(await resRaw.json());
      setProcessedImages(await resProc.json());
      setShowGallery(true);
    } catch (e) {
      Alert.alert('Connection Failed', 'Could not connect to the local API server.');
    }
  };

  const selectFromGallery = (item: any, isProcessed: boolean) => {
    setShowGallery(false);
    
    if (isProcessed) {
      const newPhoto: PhotoItem = {
        id: item.id,
        uri: item.uri,
        timestamp: Date.now(),
        analyzed: true,
        resultUri: item.uri
      };
      setSelectedPhoto(newPhoto);
      setShowCamera(false);
    } else {
      const newPhoto: PhotoItem = {
        id: item.id,
        uri: item.uri,
        timestamp: Date.now(),
        analyzed: false,
      };
      if (!photos.find(p => p.id === newPhoto.id)) {
        setPhotos(prev => [newPhoto, ...prev]);
      }
      setSelectedPhoto(newPhoto);
      setShowCamera(false);
    }
  };

  const analyzePhoto = async (photo: PhotoItem) => {
    if (isAnalyzing) return;
    setIsAnalyzing(true);
    try {
      const res = await fetch(`${API_URL}/analyze/${photo.id}`, { method: 'POST' });
      const data = await res.json();
      if (data.resultUri) {
        setPhotos(prev =>
          prev.map(p => p.id === photo.id ? { ...p, analyzed: true, resultUri: data.resultUri } : p)
        );
        setSelectedPhoto(prev => 
          prev?.id === photo.id ? { ...prev, analyzed: true, resultUri: data.resultUri } : prev
        );
      }
    } catch (e) {
      Alert.alert('Analysis Error', 'Failed to analyze the image.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (!permission) return <View style={s.container} />;

  if (!permission.granted) {
    return (
      <View style={s.container}>
        <StatusBar barStyle="light-content" />
        <View style={s.permissionBox}>
          <Ionicons name="camera" size={64} color={C.white} style={{ marginBottom: 24 }} />
          <Text style={s.permTitle}>Camera Access</Text>
          <Text style={s.permDesc}>
            eyeglassDetection needs access to your camera to detect eyeglasses and frames in real-time.
          </Text>
          <TouchableOpacity style={s.permBtn} onPress={requestPermission}>
            <Text style={s.permBtnText}>Allow Access</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const handleMainAction = () => {
    if (showCamera) {
      takePhoto();
    } else if (selectedPhoto && !selectedPhoto.analyzed) {
      analyzePhoto(selectedPhoto);
    } else {
      setShowCamera(true);
    }
  };

  return (
    <View style={s.container}>
      <StatusBar barStyle="light-content" />

      {/* ─── GALLERY MODAL (iOS Sheet Style) ─── */}
      <Modal visible={showGallery} animationType="slide" transparent={true} statusBarTranslucent>
        <View style={s.modalOverlay}>
          <TouchableOpacity 
            activeOpacity={1} 
            style={s.modalHeaderGap} 
            onPress={() => setShowGallery(false)} 
          />
          <View style={s.modalContainer}>
            <View style={s.modalHeader}>
              <View style={s.tabContainer}>
                <TouchableOpacity onPress={() => setGalleryTab('raw')} style={[s.tabBtn, galleryTab === 'raw' && s.tabBtnActive]}>
                  <Text style={[s.tabText, galleryTab === 'raw' && s.tabTextActive]}>Library</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setGalleryTab('processed')} style={[s.tabBtn, galleryTab === 'processed' && s.tabBtnActive]}>
                  <Text style={[s.tabText, galleryTab === 'processed' && s.tabTextActive]}>Analyzed</Text>
                </TouchableOpacity>
              </View>
              <TouchableOpacity onPress={() => setShowGallery(false)}>
                <Text style={s.modalActionText}>Close</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              data={galleryTab === 'raw' ? galleryImages : processedImages}
              numColumns={3}
              keyExtractor={item => item.id}
              contentContainerStyle={{ padding: 2 }}
              renderItem={({ item }) => (
                <TouchableOpacity style={s.modalThumb} onPress={() => selectFromGallery(item, galleryTab === 'processed')}>
                  <Image source={{ uri: item.uri }} style={{ width: '100%', height: '100%' }} />
                </TouchableOpacity>
              )}
            />
          </View>
        </View>
      </Modal>

      {/* ─── HEADER ─── */}
      <View style={s.header}>
        <Ionicons name="glasses-outline" size={28} color={C.white} style={{ marginRight: 8 }} />
        <Text style={s.headerTitle}>eyeglassDetection</Text>
      </View>

      {/* ─── MAIN VIEWFINDER ─── */}
      <View style={s.viewfinderOuter}>
        <View style={s.cameraWrapper}>
          {showCamera ? (
            <CameraView ref={cameraRef} style={s.camera} facing="back" />
          ) : selectedPhoto ? (
            <View style={s.camera}>
              <Image 
                source={{ uri: selectedPhoto.resultUri || selectedPhoto.uri }} 
                style={s.camera} 
                resizeMode="cover" 
              />
              
              {/* Blur Overlay during analysis */}
              {isAnalyzing && (
                <BlurView intensity={80} tint="dark" style={s.analyzeOverlay}>
                  <ActivityIndicator size="large" color={C.white} />
                  <Text style={s.analyzeText}>Analyzing Frame...</Text>
                </BlurView>
              )}

              {/* iOS style close button for photo view */}
              {!isAnalyzing && (
                <TouchableOpacity 
                  style={s.closePhotoButton} 
                  onPress={() => setShowCamera(true)}
                >
                  <BlurView intensity={60} tint="dark" style={s.closePhotoBlur}>
                    <Ionicons name="close" size={24} color={C.white} />
                  </BlurView>
                </TouchableOpacity>
              )}
            </View>
          ) : null}
        </View>
      </View>

      {/* ─── BOTTOM CONTROLS ─── */}
      <View style={s.bottomControls}>
        
        {/* Recents Thumbnail (Left) */}
        <TouchableOpacity style={s.sideButton} onPress={fetchGallery}>
          {photos.length > 0 ? (
            <Image source={{ uri: photos[0].uri }} style={s.thumbnailImage} />
          ) : (
            <View style={s.thumbnailPlaceholder}>
              <Ionicons name="images" size={24} color={C.white} />
            </View>
          )}
        </TouchableOpacity>

        {/* Shutter (Center) */}
        <Animated.View style={{ transform: [{ scale: shutterAnim }] }}>
          <TouchableOpacity 
            activeOpacity={1}
            onPressIn={handleShutterPressIn}
            onPressOut={handleShutterPressOut}
            onPress={handleMainAction}
            style={[s.shutterOuter, !showCamera && selectedPhoto && !selectedPhoto.analyzed && { borderColor: C.blue }]}
          >
            <View style={[s.shutterInner, !showCamera && selectedPhoto && !selectedPhoto.analyzed && { backgroundColor: C.blue }]} />
          </TouchableOpacity>
        </Animated.View>

        {/* Flip / Mode (Right) */}
        <TouchableOpacity style={s.sideButton}>
          <BlurView intensity={60} tint="dark" style={s.iconCircle}>
            <Ionicons name="camera-reverse" size={24} color={C.white} />
          </BlurView>
        </TouchableOpacity>

      </View>
    </View>
  );
}

const s = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: C.bg,
  },

  // ── HEADER ──
  header: {
    flexDirection: 'row',
    justifyContent: 'center',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 24,
    paddingBottom: 16,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: C.white,
    letterSpacing: -0.4,
  },

  // ── VIEWFINDER ──
  viewfinderOuter: {
    flex: 1,
    paddingHorizontal: 8,
    paddingBottom: 16,
  },
  cameraWrapper: {
    flex: 1,
    borderRadius: 32,
    overflow: 'hidden',
    backgroundColor: C.card,
  },
  camera: {
    ...StyleSheet.absoluteFillObject,
  },

  // ── PHOTO VIEW OVERLAYS ──
  analyzeOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
  },
  analyzeText: {
    color: C.white,
    fontSize: 15,
    fontWeight: '500',
  },
  closePhotoButton: {
    position: 'absolute',
    top: 24,
    left: 24,
    borderRadius: 20,
    overflow: 'hidden',
  },
  closePhotoBlur: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // ── BOTTOM CONTROLS ──
  bottomControls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-evenly',
    paddingBottom: Platform.OS === 'ios' ? 50 : 24,
    paddingTop: 10,
    paddingHorizontal: 20,
  },
  sideButton: {
    width: 50,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  iconCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  thumbnailPlaceholder: {
    width: 48,
    height: 48,
    borderRadius: 8,
    backgroundColor: C.card,
    justifyContent: 'center',
    alignItems: 'center',
  },
  thumbnailImage: {
    width: 48,
    height: 48,
    borderRadius: 8,
  },

  // Shutter Button (iOS Camera Style)
  shutterOuter: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 4,
    borderColor: C.white,
    justifyContent: 'center',
    alignItems: 'center',
  },
  shutterInner: {
    width: 62,
    height: 62,
    borderRadius: 31,
    backgroundColor: C.white,
  },

  // ── MODAL GALLERY ──
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  modalHeaderGap: {
    height: 110, // Logo ve Header'ın görüneceği boşluk
  },
  modalContainer: {
    flex: 1,
    backgroundColor: C.bg,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 16,
    borderBottomWidth: 0.5,
    borderBottomColor: C.border,
  },
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: C.card,
    borderRadius: 8,
    padding: 2,
  },
  tabBtn: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  tabBtnActive: {
    backgroundColor: '#3A3A3C',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '500',
    color: C.textSecondary,
  },
  tabTextActive: {
    color: C.white,
    fontWeight: '600',
  },
  modalTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: C.white,
  },
  modalActionText: {
    fontSize: 17,
    color: C.blue,
  },
  modalThumb: {
    width: W / 3 - 4,
    height: W / 3 - 4,
    margin: 2,
    borderRadius: 4,
    overflow: 'hidden',
    backgroundColor: C.card,
  },

  // ── PERMISSION SCREEN ──
  permissionBox: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  permTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: C.white,
    marginBottom: 12,
  },
  permDesc: {
    fontSize: 15,
    color: C.textSecondary,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 32,
  },
  permBtn: {
    backgroundColor: C.blue,
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 14,
  },
  permBtnText: {
    color: C.white,
    fontSize: 16,
    fontWeight: '600',
  },
});
