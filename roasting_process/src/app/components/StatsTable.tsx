"use client";

import React, { useState, useEffect } from "react";
import "./StatsTable.css"
interface StatsRow {
  index: string;
  [key: string]: string | number;
}

interface StatsTableProps {
  stats: StatsRow[];
}



const StatsTable: React.FC<StatsTableProps> = ({ stats }) => {
  if (!stats || stats.length === 0) return <p>No hay estadísticas disponibles.</p>;

const [selectedRow, setSelectedRow] = React.useState<number | null>(null);

  return (
    <div className = 'table' style={{ overflowX: "auto", marginTop: "20px" }}>
      <h2 className="statstitle">Estadística descriptiva de los datos</h2>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ border: "1px solid #ccc", padding: "8px" }}>Dato</th>
            {Object.keys(stats[0])
              .filter((key) => key !== "index")
              .map((col) => (
                <th key={col} style={{ border: "1px solid #ccc", padding: "8px" }}>
                  {col}
                </th>
              ))}
          </tr>
        </thead>
        <tbody>
            {stats.map((row, i) => (
                <tr
                key={i}
                onClick={() => setSelectedRow(i)}
                className={selectedRow === i ? "selected-row" : ""}
                >
                <td className="stats-cell">{row.index}</td>
                {Object.entries(row)
                    .filter(([key]) => key !== "index")
                    .map(([key, value]) => (
                    <td key={key} className="stats-cell">
                        {typeof value === "number" ? value.toFixed(2) : value}
                    </td>
                    ))}
                </tr>
            ))}
        </tbody>

      </table>
    </div>
  );
};

export default StatsTable;
