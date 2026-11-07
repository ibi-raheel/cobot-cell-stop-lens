import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/saira-semi-condensed/400.css";
import "@fontsource/saira-semi-condensed/500.css";
import "@fontsource/saira-semi-condensed/600.css";
import "@fontsource/saira-semi-condensed/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "./styles.css";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
