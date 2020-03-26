package com.example.myapplication;


import android.content.res.AssetManager;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public class kNN {
    public ArrayList<float []> data;
    public ArrayList<Long> label;
    public ArrayList<String> class_name;
    public String name;
    public double min_distance;
    public float confident;

    public kNN(AssetManager mgr){
        this.data = read_data(mgr);
        this.label = read_label(mgr);
        this.class_name = read_class(mgr);
    }
    public static double l2_norm( float[] vec1, float[] vec2){
        double temp = 0;

        for(int i = 0; i <vec1.length;i++){
            temp += Math.pow((vec1[i] - vec2[i]), 2);
        }
        temp = Math.sqrt(temp);
        return temp;
    }

    public int predict(float[] input, int K){
        ArrayList<Double> distance = new ArrayList<Double>();
        HashMap<Double, Long> map = new HashMap<>();
        HashSet<Long> set = new HashSet<Long>(label);
        int numUnique = set.size();
        int[]count = new int[numUnique];
        for(int i = 0; i < data.size();i++) {
            double tem = l2_norm(data.get(i),input);

            distance.add(tem);
            map.put(tem, label.get(i));
        }

        Collections.sort(distance);
        this.min_distance = distance.get(0);
        for(int i = 0; i < K; i++) {
            int k = 0;
            for(long j:set) {
                if(map.get(distance.get(i)) == j) {
                    count[k]++;
                }
                k++;
            }

//	        	System.out.println(map.get(distance.get(i)));
        }
        long max = 0;
        int index_max = -100;
//	        System.out.println(Collections.max(count));
        for(int i = 0; i < numUnique;i++) {
            if(max < count[i])  {
                max = count[i];
                index_max = i;
            }
        }
        this.confident = (float) count[index_max]/K;
    return index_max;
//        System.out.println(index_max);
    }
    public ArrayList<String> read_class(AssetManager mgr){
        ArrayList<String> label = new ArrayList<String>();
        JSONParser parser = new JSONParser();
//        String path = Filename + "/data.txt";
        String tContents = "";

        try {
            InputStream stream = mgr.open("class.txt");
            BufferedReader br = new BufferedReader(new InputStreamReader(stream));

            String line;
            while ((line = br.readLine()) != null) {
                label.add(line);
            }
        }
         catch (IOException e) {
            e.printStackTrace();
        }

        return label;

    }
    public ArrayList<Long> read_label(AssetManager mgr){
        ArrayList label = new ArrayList();
        JSONParser parser = new JSONParser();
//        String path = Filename + "/data.txt";
        String tContents = "";

        try {
            InputStream stream = mgr.open("data.txt");

            int size = stream.available();
            byte[] buffer = new byte[size];
            stream.read(buffer);
            stream.close();
            tContents = new String(buffer);

//            Reader reader = new FileReader(path);
            JSONObject jsonObject = (JSONObject) parser.parse(tContents);
//
//	            String name = (String) jsonObject.get("person");
//	            System.out.println(name);

            // loop array
            JSONArray msg = (JSONArray) jsonObject.get("person");
            Iterator iterator = msg.iterator();
//	            int size = Iterators.size(iterator);
            JSONParser parser1 = new JSONParser();
            while (iterator.hasNext()) {

                JSONObject emb_json = (JSONObject) parser.parse(iterator.next().toString());
                label.add(emb_json.get("name"));

            }
        } catch (IOException | ParseException e) {
            // Handle exceptions here
        }
        return label;
    }
    public ArrayList<float []> read_data(AssetManager mgr){

        JSONParser parser = new JSONParser();
        ArrayList<float[]> embedding_vector = new ArrayList<float[]>();
        String tContents = "";
        try {
            InputStream stream = mgr.open("data.txt");

            int size = stream.available();
            byte[] buffer = new byte[size];
            stream.read(buffer);
            stream.close();
            tContents = new String(buffer);

//            Reader reader = new FileReader(path);
            JSONObject jsonObject = (JSONObject) parser.parse(tContents);
            // loop array
            JSONArray msg = (JSONArray) jsonObject.get("person");
            Iterator iterator = msg.iterator();
//	            int size = Iterators.size(iterator);
            JSONParser parser1 = new JSONParser();
            while (iterator.hasNext()) {

                JSONObject emb_json = (JSONObject) parser.parse(iterator.next().toString());
                JSONArray embedding = (JSONArray) emb_json.get("emb");
                System.out.println( emb_json.get("name"));

//	            	System.out.println(label.toString());
                String floatStr = embedding.toString().replace("[", "").replace("]", "");;
                String[] valuesArr = floatStr.split(",");
                float[] floatArr = new float[valuesArr.length];

                for (int i = 0; i < valuesArr.length; i++) {
                    String floatString = valuesArr[i];

                    if (floatStr.isEmpty() ||floatStr.trim().isEmpty()) {
                        floatArr[i] = 0.0f;
                        continue;
                    }

                    floatArr[i] = Float.parseFloat(floatString.trim());

//	            	        System.out.print(floatArr[i] + " ");
                }
                embedding_vector.add(floatArr);
//	            	    System.out.println();


            }

//            System.out.println(embedding_vector.get(2).toString());
        }catch(Exception pe) {

//            System.out.println("position: " + pe.getLocalizedMessage());
//            System.out.println(pe);
        }
        return embedding_vector;
    }
}
