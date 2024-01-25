public class Square{
    double sideLength;

    public Square(double side){
        sideLength = side;
    }

    public void setSideLength(double length){
        sideLength = length;
    }

    public double getPerimeter(){
        return sideLength * 4;
    }

    public double getArea(){
        return sideLength * sideLength;
    }

}