import { NativeModule, requireNativeModule } from 'expo';

declare class EyeglassDetectorModule extends NativeModule<{}> {
  setValueAsync(value: string): Promise<void>;
}

export default requireNativeModule<EyeglassDetectorModule>('EyeglassDetector');
