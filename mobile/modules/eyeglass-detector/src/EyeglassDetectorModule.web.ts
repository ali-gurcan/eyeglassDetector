import { registerWebModule, NativeModule } from 'expo';

// EyeglassDetectorModule is not available on the web platform.
class EyeglassDetectorModule extends NativeModule<{}> {}

export default registerWebModule(EyeglassDetectorModule, 'EyeglassDetectorModule');
