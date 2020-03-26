package com.example.myapplication;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;
import java.util.ArrayList;
import java.util.Vector;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class Facenet {
    private String inputName;
    private String outputName;
    private int inputSize = 160;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private static final String MODEL_FILE  = "file:///android_asset/facenet.pb";
    //tensor name
    private static final String  FacenetInput  ="input:0";
    private static final String PhaseTrain = "phase_train:0";
    private static final String[] FacenetOutput  =new String[]{"embeddings:0"};

    private static final String TAG="facenet";
    private AssetManager assetManager;
    private TensorFlowInferenceInterface inferenceInterface;
    Facenet(AssetManager mgr){
        assetManager=mgr;
        loadModel();
    }
    private boolean loadModel() {
        //AssetManager
        try {
            inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
            Log.d("Facenet","[*]load model success");
        }catch(Exception e){
            Log.e("Facenet","[*]load model failed"+e);
            return false;
        }
        return true;
    }
    private float[] normalizeImage(Bitmap bitmap){
        int w=bitmap.getWidth();
        int h=bitmap.getHeight();
        float[] floatValues=new float[w*h*3];
        int[]   intValues=new int[w*h];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        float imageMean=127.5f;
        float imageStd=128;

        for (int i=0;i<intValues.length;i++){
            final int val=intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        return floatValues;
    }

    private Bitmap bitmapResize(Bitmap bm, float scale) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        // CREATE A MATRIX FOR THE MANIPULATION。matrix指定图片仿射变换参数
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scale, scale);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, true);
        return resizedBitmap;
    }

    protected  ArrayList <float []> regconizeFace(Bitmap bitmap){
        int w =bitmap.getWidth();
        int h =bitmap.getHeight();

        float [] input = normalizeImage(bitmap);
        boolean train = false;
        inferenceInterface.feed(FacenetInput, input, 1, inputSize, inputSize, 3);
        inferenceInterface.feed("phase_train", new boolean[]{false});
        inferenceInterface.run(FacenetOutput,false);

        float[] embeddings =new float[512];
        inferenceInterface.fetch(FacenetOutput[0],embeddings);
        ArrayList <float []> temp = new ArrayList <float []>();
        temp.add(embeddings);
        return temp;
    }

}
