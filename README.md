# Overview
StressWatch is a prototype application designed to identify different level of stress (emotional, cognitive, and physical) and relax state of the users based on physiological biosensor signals. The required signals for StressWatch to recognize the effective state of the users are 3D Accelerometer, ElectroDermal Activity (EDA), Temperature, Heart rate, Oxygen saturation level (SpO2) which can be collected through wearable (wrist) sensors. The events can ba identified using either LSTM or a more simple classifier like random forest. The associated files can be found in the notebooks folder. 

# Training Data
StressWatch detects effective state (type of stress/relax) of the users using Non-EEG Data-set for Assessment of Neurological Status that provides labeled time-series biosensor signals regarding the effective neurological state. 

# Instruction
You can access the app at http://www.stresswatch.info:8501/

The app enables the users to either view raw signals or view prediction. In the prediction mode, the top plot (line-chart) shows the transition of the effective state of users over time of the recorded signals.
 
The users can adjust the desired time period by using the time slider. The bar chart shows the respective dominant state associated with the selected/desired time period. 

