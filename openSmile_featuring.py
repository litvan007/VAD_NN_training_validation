import opensmile
import pandas

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

y = smile.process_file('/Users/litvan007/NN_sound_data_base/data_2/8f0130b5b783f5c9cd0cbc470737e917_SNR_1.wav')

print(smile.feature_names)
y.to_csv('./opensmile_ex.csv')

