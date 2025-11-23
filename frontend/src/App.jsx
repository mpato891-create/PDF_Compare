import { useState } from "react";
import {
  Container,
  Typography,
  Button,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  IconButton,
  useTheme,
  Box,
  Stack,
  Paper,
  Divider,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/CloudUpload";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import DeleteIcon from "@mui/icons-material/Delete";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import VisibilityIcon from "@mui/icons-material/Visibility";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import InfoIcon from "@mui/icons-material/Info";
import axios from "axios";

function App() {
  const theme = useTheme();
  const [standard, setStandard] = useState(null);
  const [standardText, setStandardText] = useState(""); // لعرض نص الملف الرئيسي
  const [contracts, setContracts] = useState([]);
  const [comparisons, setComparisons] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleMultipleFiles = (e) => {
    const files = Array.from(e.target.files);
    setContracts((prev) => [...prev, ...files]);
  };

  const removeContract = (index) =>
    setContracts(contracts.filter((_, i) => i !== index));

  const removeStandard = () => {
    setStandard(null);
    setStandardText("");
  };

  const handleStandardUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setStandard(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://localhost:8000/extract-preview",
        formData
      );
      setStandardText(res.data.text || "No readable text found.");
    } catch (err) {
      setStandardText("Failed to extract text from the master contract.");
    }
  };

  const handleCompareAll = async () => {
    if (!standard) return setError("Please upload the Standard Contract first");
    if (contracts.length === 0)
      return setError("Please upload at least one contract to compare");

    setLoading(true);
    setError("");
    setComparisons([]);

    const results = [];
    for (let i = 0; i < contracts.length; i++) {
      const formData = new FormData();
      formData.append("standard", standard);
      formData.append("other", contracts[i]);

      try {
        const res = await axios.post("http://localhost:8000/compare", formData);
        results.push({
          name: contracts[i].name,
          report: res.data.report,
          success: true,
        });
      } catch (err) {
        results.push({
          name: contracts[i].name,
          report:
            "Failed to compare this contract (server error or corrupted file)",
          success: false,
        });
      }
    }
    setComparisons(results);
    setLoading(false);
  };

  // دالة التقرير الملون
  const renderBeautifulReport = (report) => {
    if (!report || report.includes("Failed") || report.includes("خطأ")) {
      return (
        <Alert severity="error" sx={{ mt: 2 }}>
          <strong>Comparison Failed:</strong> {report}
        </Alert>
      );
    }

    const lines = report
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    const sections = { missing: [], modified: [], additional: [], summary: [] };
    let currentSection = null;

    lines.forEach((line) => {
      if (line.includes("Missing Clauses")) currentSection = "missing";
      else if (line.includes("Modified Clauses")) currentSection = "modified";
      else if (line.includes("Additional Clauses"))
        currentSection = "additional";
      else if (line.includes("Summary") || line.includes("Risk Level"))
        currentSection = "summary";
      else if (
        (line.startsWith("-") || line.startsWith("•")) &&
        currentSection
      ) {
        sections[currentSection].push(line.slice(1).trim());
      } else if (currentSection === "summary") {
        sections.summary.push(line);
      }
    });

    const riskMatch = report.match(/Risk Level[:\s]+\*\*(.+?)\*\*/i);
    const riskLevel = riskMatch ? riskMatch[1].trim() : "Low";

    const riskColor =
      {
        Critical: "#d32f2f",
        High: "#f57c00",
        Medium: "#ff9800",
        Low: "#4caf50",
      }[riskLevel] || "#666";

    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ textAlign: "center", mb: 4 }}>
          <Chip
            label={`Risk Level: ${riskLevel}`}
            sx={{
              fontSize: "1.5rem",
              fontWeight: "bold",
              py: 3.5,
              px: 6,
              backgroundColor: riskColor,
              color: "white",
              boxShadow: "0 8px 25px rgba(0,0,0,0.3)",
            }}
          />
        </Box>

        <Stack spacing={4}>
          <Box>
            <Typography
              variant="h6"
              color="error"
              sx={{ display: "flex", alignItems: "center", gap: 1 }}
            >
              <WarningAmberIcon /> Missing Clauses ({sections.missing.length})
            </Typography>
            {sections.missing.length > 0 ? (
              <Stack spacing={1.5} sx={{ mt: 2, pl: 3 }}>
                {sections.missing.map((item, i) => (
                  <Typography
                    key={i}
                    color="error.dark"
                    sx={{ fontWeight: 500 }}
                  >
                    • {item}
                  </Typography>
                ))}
              </Stack>
            ) : (
              <Typography color="success.main" sx={{ pl: 3, mt: 1 }}>
                No missing clauses detected
              </Typography>
            )}
          </Box>

          <Divider />

          <Box>
            <Typography variant="h6" color="#ff8f00">
              Modified Clauses ({sections.modified.length})
            </Typography>
            {sections.modified.length > 0 ? (
              <Stack spacing={1.5} sx={{ mt: 2, pl: 3 }}>
                {sections.modified.map((item, i) => (
                  <Alert key={i} severity="warning" variant="outlined">
                    {item}
                  </Alert>
                ))}
              </Stack>
            ) : (
              <Typography color="success.main" sx={{ pl: 3, mt: 1 }}>
                No modifications detected
              </Typography>
            )}
          </Box>

          <Divider />

          <Box>
            <Typography
              variant="h6"
              color="primary"
              sx={{ display: "flex", alignItems: "center", gap: 1 }}
            >
              <InfoIcon /> Additional Clauses ({sections.additional.length})
            </Typography>
            {sections.additional.length > 0 ? (
              <Stack spacing={1.5} sx={{ mt: 2, pl: 3 }}>
                {sections.additional.map((item, i) => (
                  <Typography key={i} color="primary.dark">
                    • {item}
                  </Typography>
                ))}
              </Stack>
            ) : (
              <Typography color="text.secondary" sx={{ pl: 3, mt: 1 }}>
                No extra clauses found
              </Typography>
            )}
          </Box>

          <Divider />

          <Box>
            <Typography variant="h6" gutterBottom>
              Summary
            </Typography>
            <Paper
              elevation={3}
              sx={{
                p: 3,
                bgcolor: "#f5f5f5",
                fontFamily: "monospace",
                fontSize: "0.95rem",
              }}
            >
              {sections.summary.map((line, i) => (
                <Typography key={i}>{line}</Typography>
              ))}
            </Paper>
          </Box>
        </Stack>
      </Box>
    );
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        backgroundAttachment: "fixed",
        py: { xs: 4, md: 8 },
      }}
    >
      <Container maxWidth="lg">
        <Typography
          variant="h3"
          align="center"
          fontWeight="bold"
          sx={{
            color: "white",
            mb: 2,
            fontSize: { xs: "2.5rem", md: "4rem" },
            textShadow: "0 4px 20px rgba(0,0,0,0.3)",
          }}
        >
          Smart Contract Comparator
        </Typography>
        <Typography
          variant="h6"
          align="center"
          sx={{
            color: "rgba(255,255,255,0.9)",
            mb: 6,
            maxWidth: "800px",
            mx: "auto",
          }}
        >
          Upload your master contract once, then compare unlimited contracts
          instantly with advanced analysis
        </Typography>

        <Stack spacing={4}>
          {/* Master Contract مع Preview */}
          <Paper
            elevation={12}
            sx={{
              p: 4,
              borderRadius: 4,
              backdropFilter: "blur(10px)",
              background: "rgba(255,255,255,0.15)",
              border: "1px solid rgba(255,255,255,0.2)",
            }}
          >
            <Typography
              variant="h5"
              gutterBottom
              color="white"
              fontWeight="bold"
            >
              Master Contract (Standard Template)
            </Typography>

            {!standard ? (
              <Button
                variant="contained"
                component="label"
                startIcon={<UploadFileIcon />}
                fullWidth
                sx={{
                  py: 2,
                  background:
                    "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
                  color: "white",
                }}
              >
                Upload Master Contract
                <input
                  type="file"
                  hidden
                  accept=".pdf,.docx"
                  onChange={handleStandardUpload}
                />
              </Button>
            ) : (
              <Box>
                <Box
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  mb={2}
                >
                  <Chip
                    label={standard.name}
                    sx={{
                      background:
                        "linear-gradient(45deg, #4ECDC4 30%, #44A08D 90%)",
                      color: "white",
                      fontWeight: "bold",
                      py: 3,
                    }}
                  />
                  <IconButton onClick={removeStandard} color="error">
                    <DeleteIcon />
                  </IconButton>
                </Box>

                <Accordion
                  sx={{
                    bgcolor: "rgba(255,255,255,0.95)",
                    color: "black",
                    borderRadius: 3,
                    "&:before": { display: "none" },
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography
                      fontWeight="bold"
                      color="primary"
                      sx={{ display: "flex", alignItems: "center", gap: 1 }}
                    >
                      <VisibilityIcon /> Preview Master Contract Text
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box
                      sx={{
                        maxHeight: 400,
                        overflow: "auto",
                        p: 2,
                        bgcolor: "#fafafa",
                        borderRadius: 2,
                        fontFamily: "monospace",
                        fontSize: "0.9rem",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {standardText || "Extracting text..."}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Box>
            )}
          </Paper>

          {/* Contracts to Compare */}
          <Paper
            elevation={12}
            sx={{
              p: 4,
              borderRadius: 4,
              backdropFilter: "blur(10px)",
              background: "rgba(255,255,255,0.15)",
              border: "1px solid rgba(255,255,255,0.2)",
            }}
          >
            <Typography
              variant="h5"
              gutterBottom
              color="white"
              fontWeight="bold"
            >
              Contracts to Compare ({contracts.length})
            </Typography>
            <Button
              variant="outlined"
              component="label"
              startIcon={<UploadFileIcon />}
              fullWidth
              sx={{
                border: "2px dashed rgba(255,255,255,0.5)",
                color: "white",
                py: 3,
                mb: 3,
              }}
            >
              Drop or Select Multiple Contracts
              <input
                type="file"
                hidden
                multiple
                accept=".pdf,.docx"
                onChange={handleMultipleFiles}
              />
            </Button>
            <Stack spacing={2}>
              {contracts.map((file, i) => (
                <Box
                  key={i}
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    p: 2,
                    bgcolor: "rgba(255,255,255,0.1)",
                    borderRadius: 2,
                  }}
                >
                  <Typography color="white" noWrap flex={1}>
                    {file.name}
                  </Typography>
                  <IconButton onClick={() => removeContract(i)} color="error">
                    <DeleteIcon />
                  </IconButton>
                </Box>
              ))}
            </Stack>
          </Paper>

          <Button
            variant="contained"
            size="large"
            startIcon={<CompareArrowsIcon />}
            onClick={handleCompareAll}
            disabled={loading || !standard || contracts.length === 0}
            fullWidth
            sx={{
              py: 3,
              fontSize: "1.4rem",
              fontWeight: "bold",
              background: "linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)",
            }}
          >
            {loading ? "Analyzing Contracts..." : "Compare All Contracts Now"}
          </Button>

          {loading && <LinearProgress sx={{ height: 8, borderRadius: 4 }} />}
          {error && <Alert severity="error">{error}</Alert>}

          {/* Results */}
          {comparisons.length > 0 && (
            <Box>
              <Typography
                variant="h4"
                color="white"
                gutterBottom
                fontWeight="bold"
              >
                Comparison Results
              </Typography>
              {comparisons.map((result, i) => (
                <Accordion
                  key={i}
                  elevation={10}
                  sx={{
                    mb: 3,
                    borderRadius: 3,
                    bgcolor: "rgba(255,255,255,0.97)",
                  }}
                >
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography
                      fontWeight="bold"
                      color={result.success ? "success.main" : "error.main"}
                    >
                      {result.name} → {result.success ? "Success" : "Failed"}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    {renderBeautifulReport(result.report)}
                  </AccordionDetails>
                </Accordion>
              ))}
            </Box>
          )}
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
