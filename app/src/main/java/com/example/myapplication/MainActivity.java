package com.example.myapplication;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.ImageReader;
import android.os.SystemClock;
import android.util.Size;

import com.example.myapplication.env.ImageUtils;
import com.example.myapplication.env.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import com.example.myapplication.env.BorderedText;
import com.example.myapplication.env.ImageUtils;
import com.example.myapplication.env.Logger;
import com.example.myapplication.tracking.MultiBoxTracker;

public class MainActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    MTCNN mtcnn;
    Facenet facenet;
    kNN knn;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;


    private Integer sensorOrientation;

    private static final int INPUT_SIZE = 500;
    private static final boolean MAINTAIN_ASPECT = true;
    private boolean computingDetection = false;
    private byte[] luminanceCopy;

    private MultiBoxTracker tracker;
    OverlayView trackingOverlay;

    @Override
    protected void processImage() {

        byte[] originalLuminance = getLuminance();

        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance);
        trackingOverlay.postInvalidate();

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);



        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);

        readyForNextImage();

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        Bitmap face=null;
                        Bitmap face1=null;
                        ArrayList <RectF> results = new ArrayList<RectF>();

                        ArrayList <Regconition> reg = new ArrayList<Regconition>();

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);
//                        final long startTime = SystemClock.uptimeMillis();
                        Vector<Box> boxes=mtcnn.detectFaces(croppedBitmap,40);
                        for (int i=0;i<boxes.size();i++) {

                            RectF location = boxes.get(i).transform2RectF();
                            canvas.drawRect(boxes.get(i).transform2Rect(), paint);


                            int [] box = boxes.get(i).box;
                            try {
                                face = Bitmap.createBitmap(cropCopyBitmap, box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1);
                                face1 = Bitmap.createScaledBitmap(face, 160, 160, false);
                                ArrayList<float []> temp = facenet.regconizeFace(face1);
                                String name;
                                int class_name = knn.predict(temp.get(0),5);
                                if(knn.min_distance < 2) {
                                    name = knn.class_name.get(class_name);
                                }
                                else{
                                    name = "unknown";
                                }

                                cropToFrameTransform.mapRect(location);
                                results.add(location);
                                reg.add(new Regconition(name,knn.confident,location));
                            }
                            catch(Exception e){

                            }

                        }
                        tracker.trackResults(reg, luminanceCopy);
                        trackingOverlay.postInvalidate();
                        requestRender();

                    }
                });

    }

    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        AssetManager asm=getAssets();

        mtcnn=new MTCNN(getAssets());
        facenet = new Facenet(getAssets());

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);

        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                INPUT_SIZE, INPUT_SIZE,
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                    }
                });

        try{
            knn = new kNN(getAssets());
//            svm.train("-t 2 "/* svm kernel */ + appFolderPath+ "heart_scale " +appFolderPath+ "model");
//            svm.predict(appFolderPath + "heart_scale_predict " + appFolderPath + "model " + appFolderPath + "result");
//            jniSvmPredict(appFolderPath + "heart_scale_predict " + " " + appFolderPath + "model " + " " +appFolderPath + "result");

        } catch (Exception e) {
            e.printStackTrace();
        }

        addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        if (!isDebug()) {
                            return;
                        }
                        final Bitmap copy = cropCopyBitmap;
                        if (copy == null) {
                            return;
                        }

                        final int backgroundColor = Color.argb(100, 0, 0, 0);
                        canvas.drawColor(backgroundColor);

                        final Matrix matrix = new Matrix();
                        final float scaleFactor = 2;
                        matrix.postScale(scaleFactor, scaleFactor);
                        matrix.postTranslate(
                                canvas.getWidth() - copy.getWidth() * scaleFactor,
                                canvas.getHeight() - copy.getHeight() * scaleFactor);
                        canvas.drawBitmap(copy, matrix, new Paint());

                        final Vector<String> lines = new Vector<String>();
//                        if (detector != null) {
//                            final String statString = detector.getStatString();
//                            final String[] statLines = statString.split("\n");
//                            for (final String line : statLines) {
//                                lines.add(line);
//                            }
//                        }
                        lines.add("");

                        lines.add("Frame: " + previewWidth + "x" + previewHeight);
                        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                        lines.add("Rotation: " + sensorOrientation);
//                        lines.add("Inference time: " + lastProcessingTimeMs + "ms");
//
//                        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                    }
                });
    }


    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }
}
