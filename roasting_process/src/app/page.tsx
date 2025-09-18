"use client";

import React, { useState, useEffect } from "react";
import GraphContainer from "./components/graphs";
import ImageCarousel from "./components/carousel";
import StatsTable from "./components/StatsTable";
import { Button } from "@/components/ui/stateful-button";
import { Buttn } from "@/components/ui/moving-border";
import "./dashboard.css";
import Separator from "./components/separator";


// Dashboard Component
const Dashboard: React.FC = () => {
  const [xColumns, setXColumns] = useState<string[]>([]);
  const [yColumns, setYColumns] = useState<string[]>([]);
  const [inputs, setInputs] = useState<number[]>([]);
  const [outputs, setOutputs] = useState<number[]>([]);
  const [targetVariable, setTargetVariable] = useState<string>("");
  const [weight, setWeight] = useState<number>(0);
  const [timestamp, setTimestamp] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [startVisible, setStartVisible] = useState(true);
  const [finalPressure, setFinalPressure] = useState<number>(0);

  interface StatsRow {
    index: string;
    [key: string]: string | number;
  }

  const [stats, setStats] = useState<StatsRow[]>([]);

  // Fetch column names from backend on initial load
  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/columns");
        const data = await response.json();
        setXColumns(data.X_columns);
        setYColumns(data.Y_columns);
        setInputs(Array(data.X_columns.length).fill(0));
        setOutputs(Array(data.Y_columns.length).fill(0));
        setTargetVariable(data.Y_columns[0] || "");
        
        setTimestamp(Date.now());
      } catch (error) {
        console.error("Error loading columns:", error);
      }
    };

    fetchColumns();
  }, []);

  // Handle input change
  const handleChange = (index: number, value: string) => {
    const parsed = parseFloat(value);
    const newInputs = [...inputs];
    newInputs[index] = isNaN(parsed) ? 0 : parsed;
    setInputs(newInputs);
  };

  // Submit data to backend
  const handleSubmit = async () => {
    setLoading(true);
    setDone(false);


    
    if (weight <= 0 || isNaN(weight)) {
        alert("Please enter a valid weight greater than 0.");
        setLoading(false);
        return;
      }


    
    try {
      const response = await fetch("http://127.0.0.1:8000/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs, targetVariable, weight}),
      });

      if (!response.ok) {
        throw new Error(`Request error: ${response.status}`);
      }

      const data = await response.json();

      if (Array.isArray(data["results"]) && data["results"].length === yColumns.length) {
        setOutputs(data["results"]);
        setStats(data["stats"]);
        setFinalPressure(data["final_pressure"]);
        console.log(data["final_pressure"])
        setTimestamp(Date.now());
        setDone(true);
      } else {
        console.warn("Unexpected results format:", data["results"]);
      }
    } catch (error: any) {
      console.error("Error submitting data:", error.message);
    } finally {
      setLoading(false);
    }
  };

  // Custom positions for output cards
  const customCardPositions = [
    { top: "10px", left: "40px" },
    { top: "270px", left: "730px" },
    { top: "400px", left: "400px" },
    { top: "750px", left: "400px" },
    { top: "745px", left: "800px" },
    { top: "110px", left: "40px" }
  ];



  return (
    <div className="dashboard">
      {/* ================= HEADER ================= */}
      <header className="header">
        <img src="/gmcon.png" alt="Logo" />
        <h1>Roasting - Acid Plant</h1>
      </header>

      {/* ================= START SCREEN ================= */}
      <div className={`start ${startVisible ? "visible" : "hidden"}`}>
        <div className="dark-overlay"></div>
        <div className="overlay-texto">
          <p className="p1">Roasting Process</p>
          <p className="p2">Simulator</p>
          <p className="description">
            Simulate and analyze metallurgical roasting with precision and control.
          </p>
          <div>
            <Buttn
              text="Get Started"
              onClick={() => setStartVisible(false)}
              borderRadius="1.75rem"
              className="hover:cursor-pointer bg-white dark:bg-slate-900 text-black dark:text-white border-neutral-200 dark:border-slate-800"
            />
          </div>
        </div>
        <div className="final-title">Roasting Process Simulator</div>
      </div>

      {/* ================= MAIN CONTENT ================= */}
      {!startVisible && (
        <main className="dashboard-content">
          <div className="inout">
            {/* ===== INPUT SECTION ===== */}
            <section className="in">
              <h2 className="section-title">Inputs</h2>
              <div className="input-grid">
                {xColumns.map((label, index) => (
                  <div key={index} className="input-card">
                    <label className="input-label">{label}</label>
                    <input
                      type="number"
                      value={inputs[index]}
                      onChange={(e) => handleChange(index, e.target.value)}
                      className="input-field"
                    />
                  </div>
                ))}
              </div>

              {/* ===== TARGET VARIABLE SELECTOR ===== */}
              <section className="target-variable-input">
                <label className="input-label">Target Variable</label>
                <select
                  value={targetVariable}
                  onChange={(e) => setTargetVariable(e.target.value)}
                  className="input-field"
                >
                  {yColumns.map((variable, index) => (
                    <option key={index} value={variable}>
                      {variable}
                    </option>
                  ))}
                </select>
              </section>


              {/*WEIGHT */}
              <section className="weight-input">
                <label className="input-label">Weight (Ton)</label>
                <input
                  type="number"
                  value={weight}
                  onChange={(e) => setWeight(parseFloat(e.target.value))}
                  className="input-field"
                />
              </section>

              {/* ===== SUBMIT BUTTON ===== */}
              <div className="submit-container">
                <Button onClick={handleSubmit}>Start</Button>
              </div>
            </section>

            {/* ===== OUTPUT SECTION ===== */}
            <section className="roaster">
              <img src="roaster.jpg" alt="Roaster" />
              <div className="output-grid">
                {yColumns.map((label, index) => (
                  <div
                    key={index}
                    className="output-card"
                    style={customCardPositions[index]}
                  >
                    <span className="output-label">{label}</span>
                    <span>{outputs[index]?.toFixed(2) ?? "0.00"}</span>
                  </div>
                ))}
                
                <div className="output-card" style={{ top: "200px", left: "40px" }}>
                  <span className="output-label">Final Pressure</span>
                  <span>{finalPressure ? finalPressure.toFixed(2) : "0.00"} mm H2O</span>
                </div>



              </div>
            </section>
          </div>
                
          {/* ===== MODEL PERFORMANCE SECTION== */}
          <Separator text= ".   : Model Performance : ." className = "separator"></Separator>

          {/* ===== GRAPHS SECTION ===== */}
          {timestamp && (
            <section className="graph_1_container">
              {[
                ["Results vs Predictions", `/images/1_resultados_vs_predicciones.png?ts=${timestamp}`],
                ["Residuals vs Predictions", `/images/3_residuos_vs_prediccion.png?ts=${timestamp}`],
              ].map(([title, src], i) => (
                <GraphContainer
                  key={i}
                  title={title}
                  src={src}
                  alt={title}
                  width="100%"
                />
              ))}
            </section>
          )}



          {/* ===== DATA ANALYSIS SECTION ===== */}
            <Separator text=". : Data Analysis : ." className="separator" />

          <div className="data_analysis_graph_container">

            <div style={{ display: 'flex', justifyContent: 'center', margin: '2rem 0' }}>
              <img src={`/images/7_presion_vs_delta_presion.png?ts=${timestamp}`} alt="imagen" />
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', margin: '2rem 0' }}>
              <img src={`/images/8_calcine_plot.png?ts=${timestamp}`} alt="imagen" />
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', margin: '2rem 0' }}>
              <img src={`/images/9_Throat_Bed_Temperature.png?ts=${timestamp}`} alt="imagen" />
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', margin: '2rem 0' }}>
              <img src={`/images/10_Oxygen_plot.png?ts=${timestamp}`} alt="imagen" />
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', margin: '2rem 0' }}>
              <img src={`/images/11_Concentrate_plot.png?ts=${timestamp}`} alt="imagen" />
            </div>

          </div>

            
            <StatsTable stats={stats} />

          {/* ===== IMAGE CAROUSEL ===== */}

     
          <p className="text-red-500 text-2xl font-bold text-center my-8">
            Click On Images
          </p>
          <ImageCarousel />




          {/* ===== DISCLAIMER ===== */}
          <section className="note-box">
            <p className="note-title">Important Note:</p>
            <p>
              The values shown are simulated results for demonstration purposes and do not represent real data.
            </p>
          </section>
        </main>
      )}

      {/* ================= FOOTER ================= */}
      <footer className="footer">
        <img src="./rcon.png" alt="icon" className="icon-footer" />
        © Metallurgy Department, Pilot Plant
      </footer>
    </div>
  );
};

export default Dashboard;


