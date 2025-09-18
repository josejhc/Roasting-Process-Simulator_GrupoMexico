import React from 'react';
import './graphs.css';
const GraphContainer = ({ src, alt, title, width }) => {
  return (
    <div className="graph-container">
      <h3 className='title'>{title}</h3>
      <div className='graph'>
          <img src={src} alt={alt} width={width}  />
      </div>
      
    </div>
  );
};

export default GraphContainer;

