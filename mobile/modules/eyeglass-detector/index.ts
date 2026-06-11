import { requireNativeModule } from 'expo-modules-core';

const EyeglassDetector = requireNativeModule('EyeglassDetector');

export async function analyzeImage(imageUri: string): Promise<string> {
  return await EyeglassDetector.analyzeImage(imageUri);
}
