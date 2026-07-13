import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { AppConfigProvider } from "./AppConfigContext";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppConfigProvider>
      <App />
    </AppConfigProvider>
  </StrictMode>,
);
