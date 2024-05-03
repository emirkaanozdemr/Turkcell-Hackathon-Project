function showModelDescription(modelName) {
    const descriptions = {
        "Yüz Modeli": "Yüz ifadelerini analiz eden model.",
        "Konuşma Modeli": "Konuşma duygusunu algılayan model.",
        "Müzik Kümeleme Modeli": "Farklı türdeki müzikleri kümelendiren model."
    };
    alert("Seçilen Model: " + descriptions[modelName]);
}