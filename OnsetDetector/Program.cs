using AForge.Math;
using NAudio.Wave;

using (AudioFileReader reader = new AudioFileReader("./Ressources/ff123_kraftwerk.wav"))
{
    var mono = reader.ToMono();

    int fps = 70;

    Complex[] Array = new Complex[reader.Length/2];
    /*******fill the array************/
    FourierTransform.FFT(Array, FourierTransform.Direction.Forward);

    WaveFileWriter.CreateWaveFile16("Test.wav", mono);
}