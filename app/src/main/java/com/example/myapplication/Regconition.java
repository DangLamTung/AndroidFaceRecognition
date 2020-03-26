package com.example.myapplication;

import android.graphics.RectF;

public class Regconition {
    public String name;
    public float confident;
    public RectF box;
public Regconition(String name, float confident, RectF box){
    this.box = box;
    this.confident = confident;
    this.name = name;
}
}
