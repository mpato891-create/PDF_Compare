import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

import { CssBaseline } from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#1976d2" },
    secondary: { main: "#9c27b0" },
  },
  typography: { fontFamily: "Roboto, Arial, sans-serif" },
  direction: "rtl", // مهم للعربي
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* ده بيعمل reset كامل وأحسن من أي index.css */}
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
