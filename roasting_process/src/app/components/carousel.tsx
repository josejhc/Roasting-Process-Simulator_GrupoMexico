"use client";

import { useEffect, useState } from "react";

const ImageCarousel = () => {
  const [timestamp, setTimestamp] = useState<number | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Generate timestamp only on the client to avoid hydration mismatch
  useEffect(() => {
    setTimestamp(Date.now());
  }, []);

  // Define images with timestamp once it's available
  const images = timestamp
    ? [
        ["Matriz de Correlación", `/images/5_matriz_correlacion.png?ts=${timestamp}`],
        ["Histograma de Errores", `/images/4_histograma_errores.png?ts=${timestamp}`],
        ["Correlación Variable Objetivo", `/images/2_correlacion_variable_objetivo.png?ts=${timestamp}`],
        ["Importancia de Características", `/images/6_importancia_caracteristicas.png?ts=${timestamp}`],
      ]
    : [];

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const bounds = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - bounds.left;
    if (clickX < bounds.width / 2) {
      setCurrentIndex(currentIndex === 0 ? images.length - 1 : currentIndex - 1);
    } else {
      setCurrentIndex((currentIndex + 1) % images.length);
    }
  };

  if (!timestamp) return null; // Avoid rendering until timestamp is ready

  return (
    <div className="carousel-wrapper">
      <p className="carousel-title">{images[currentIndex][0]}</p>
      <div className="carousel-image-container" onClick={handleClick}>
        <img
          src={images[currentIndex][1]}
          alt={images[currentIndex][0]}
          className="carousel-image"
        />
        <div className="carousel-overlay left"></div>
        <div className="carousel-overlay right"></div>
      </div>
    </div>
  );
};

export default ImageCarousel;
