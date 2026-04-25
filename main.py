from train import train_model

if __name__ == "__main__":
    model = train_model()
    
    results = model.recommend(
        subject="Machine Learning",
        level="Beginner",
        type_="Video"
    )
    
    print(results)
