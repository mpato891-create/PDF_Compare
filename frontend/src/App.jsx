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
  Box,
  Stack,
  Paper,
  Divider,
  CircularProgress,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/CloudUpload";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import DeleteIcon from "@mui/icons-material/Delete";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import VisibilityIcon from "@mui/icons-material/Visibility";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import InfoIcon from "@mui/icons-material/Info";
import TranslateIcon from "@mui/icons-material/Translate";
import axios from "axios";

const API_BASE_URL = "https://zainmustafa-api-ai.hf.space";

function App() {
  const [standard, setStandard] = useState(null);
  const [standardText, setStandardText] = useState("");
  const [contracts, setContracts] = useState([]);
  const [comparisons, setComparisons] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [translatingIndex, setTranslatingIndex] = useState(null);

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
    setStandardText("Extracting text, please wait...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE_URL}/extract-preview`, formData);
      setStandardText(res.data.text || "No readable text found.");
    } catch (err) {
      console.error("Extraction Error:", err);
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
        const res = await axios.post(`${API_BASE_URL}/compare`, formData);
        results.push({
          name: contracts[i].name,
          report: res.data.report,
          translatedReport: null,
          isTranslated: false,
          success: true,
        });
      } catch (err) {
        console.error("Comparison Error:", err);
        results.push({
          name: contracts[i].name,
          report:
            "Failed to compare this contract (server error or corrupted file)",
          translatedReport: null,
          isTranslated: false,
          success: false,
        });
      }
    }
    setComparisons(results);
    setLoading(false);
  };

  const handleTranslateReport = async (index) => {
    setTranslatingIndex(index);
    const currentComparison = comparisons[index];

    try {
      const response = await axios.post(`${API_BASE_URL}/translate-report`, {
        text: currentComparison.report,
      });

      const updatedComparisons = [...comparisons];
      updatedComparisons[index] = {
        ...currentComparison,
        translatedReport: response.data.translated_text,
        isTranslated: true,
      };
      setComparisons(updatedComparisons);
    } catch (err) {
      console.error("Translation Error:", err);
      setError("فشل في ترجمة التقرير. حاول مرة أخرى.");
    } finally {
      setTranslatingIndex(null);
    }
  };

  const renderBeautifulReport = (report, isArabic = false) => {
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

    const sections = {
      missing: [],
      modified: [],
      additional: [],
      summary: [],
      riskLevel: "Low",
    };

    let currentSection = null;

    // البحث عن Risk Level في كامل التقرير أولاً (أكثر دقة)
    const fullReport = report.toLowerCase();
    let detectedRisk = "Low";

    // نمط 1: البحث عن "Risk Level: Critical"
    if (fullReport.includes("risk level")) {
      const patterns = [
        /risk\s*level[:\s]*\*\*\s*([a-z]+)\s*\*\*/i,
        /risk\s*level[:\s]*([a-z]+)/i,
        /\*\*risk\s*level[:\s]*([a-z]+)\*\*/i,
      ];

      for (const pattern of patterns) {
        const match = report.match(pattern);
        if (match && match[1]) {
          detectedRisk = match[1].trim();
          break;
        }
      }
    }

    // نمط 2: البحث عن "Critical/High/Medium deviations"
    if (fullReport.includes("critical deviation")) {
      detectedRisk = "Critical";
    } else if (
      fullReport.includes("high deviation") ||
      fullReport.includes("high risk")
    ) {
      detectedRisk = "High";
    } else if (
      fullReport.includes("medium deviation") ||
      fullReport.includes("moderate risk")
    ) {
      detectedRisk = "Medium";
    }

    // نمط 3: البحث عن Match Percentage
    const matchPercentMatch = report.match(/match\s*percentage[:\s]*(\d+)%/i);
    if (matchPercentMatch) {
      const percentage = parseInt(matchPercentMatch[1]);
      if (percentage < 60) {
        detectedRisk = "Critical";
      } else if (percentage < 75) {
        detectedRisk = "High";
      } else if (percentage < 90) {
        detectedRisk = "Medium";
      }
    }

    sections.riskLevel = detectedRisk;

    // نمط محسّن للتعرف على الأقسام
    lines.forEach((line) => {
      const lowerLine = line.toLowerCase();

      // تحديد القسم الحالي
      if (
        lowerLine.includes("missing clause") ||
        lowerLine.includes("البنود المفقودة") ||
        lowerLine.includes("بنود مفقودة") ||
        lowerLine.includes("⚠️") ||
        lowerLine.includes("❌")
      ) {
        currentSection = "missing";
      } else if (
        lowerLine.includes("modified clause") ||
        lowerLine.includes("البنود المعدلة") ||
        lowerLine.includes("بنود معدلة") ||
        lowerLine.includes("🔄")
      ) {
        currentSection = "modified";
      } else if (
        lowerLine.includes("additional clause") ||
        lowerLine.includes("البنود الإضافية") ||
        lowerLine.includes("بنود إضافية") ||
        lowerLine.includes("➕")
      ) {
        currentSection = "additional";
      } else if (
        lowerLine.includes("summary") ||
        lowerLine.includes("الملخص") ||
        lowerLine.includes("ملخص")
      ) {
        currentSection = "summary";
      } else if (currentSection) {
        // إضافة المحتوى للقسم الحالي
        let cleanLine = line;

        // تنظيف البداية من الرموز
        if (
          line.startsWith("-") ||
          line.startsWith("•") ||
          line.startsWith("*")
        ) {
          cleanLine = line.slice(1).trim();
        }

        // تنظيف الأرقام في البداية (1. 2. 3.)
        cleanLine = cleanLine.replace(/^\d+\.\s*/, "");

        // تنظيف النجوم ** **
        cleanLine = cleanLine.replace(/\*\*/g, "");

        // تجاهل الأسطر الفارغة أو العناوين المكررة
        if (
          cleanLine &&
          !cleanLine.toLowerCase().includes("missing clause") &&
          !cleanLine.toLowerCase().includes("modified clause") &&
          !cleanLine.toLowerCase().includes("additional clause") &&
          !cleanLine.toLowerCase().includes("البنود") &&
          !cleanLine.toLowerCase().includes("summary")
        ) {
          sections[currentSection].push(cleanLine);
        }
      }
    });

    // تحويل مستوى المخاطر للعربية إذا لزم الأمر
    const riskMapping = {
      critical: "حرج",
      high: "عالي",
      medium: "متوسط",
      low: "منخفض",
      حرج: "حرج",
      عالي: "عالي",
      متوسط: "متوسط",
      منخفض: "منخفض",
    };

    const normalizedRisk = sections.riskLevel.toLowerCase();
    const riskLevelDisplay = isArabic
      ? riskMapping[normalizedRisk] || sections.riskLevel
      : sections.riskLevel;

    const riskColorMapping = {
      critical: "#d32f2f",
      حرج: "#d32f2f",
      high: "#f57c00",
      عالي: "#f57c00",
      medium: "#ff9800",
      متوسط: "#ff9800",
      low: "#4caf50",
      منخفض: "#4caf50",
    };

    const riskColor =
      riskColorMapping[normalizedRisk] ||
      riskColorMapping[sections.riskLevel] ||
      "#4caf50";

    return (
      <Box sx={{ p: 3, direction: isArabic ? "rtl" : "ltr" }}>
        <Box sx={{ textAlign: "center", mb: 4 }}>
          <Chip
            label={
              isArabic
                ? `مستوى المخاطر: ${riskLevelDisplay}`
                : `Risk Level: ${riskLevelDisplay}`
            }
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
          {/* Missing Clauses */}
          <Box>
            <Typography
              variant="h6"
              color="error"
              sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}
            >
              <WarningAmberIcon />
              {isArabic ? "البنود المفقودة" : "Missing Clauses"} (
              {sections.missing.length})
            </Typography>
            {sections.missing.length > 0 ? (
              <Paper elevation={2} sx={{ p: 2, bgcolor: "#ffebee" }}>
                <Stack spacing={1.5}>
                  {sections.missing.map((item, i) => (
                    <Box
                      key={i}
                      sx={{
                        p: 2,
                        bgcolor: "white",
                        borderRadius: 2,
                        borderLeft: "4px solid #d32f2f",
                      }}
                    >
                      <Typography color="error.dark" sx={{ fontWeight: 500 }}>
                        {item}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              </Paper>
            ) : (
              <Alert severity="success">
                {isArabic
                  ? "✓ لا توجد بنود مفقودة"
                  : "✓ No missing clauses detected"}
              </Alert>
            )}
          </Box>

          <Divider />

          {/* Modified Clauses */}
          <Box>
            <Typography
              variant="h6"
              color="warning.dark"
              sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}
            >
              <InfoIcon />
              {isArabic ? "البنود المعدلة" : "Modified Clauses"} (
              {sections.modified.length})
            </Typography>
            {sections.modified.length > 0 ? (
              <Paper elevation={2} sx={{ p: 2, bgcolor: "#fff8e1" }}>
                <Stack spacing={1.5}>
                  {sections.modified.map((item, i) => (
                    <Alert
                      key={i}
                      severity="warning"
                      sx={{
                        bgcolor: "white",
                        "& .MuiAlert-message": { width: "100%" },
                      }}
                    >
                      {item}
                    </Alert>
                  ))}
                </Stack>
              </Paper>
            ) : (
              <Alert severity="success">
                {isArabic ? "✓ لا توجد تعديلات" : "✓ No modifications detected"}
              </Alert>
            )}
          </Box>

          <Divider />

          {/* Additional Clauses */}
          <Box>
            <Typography
              variant="h6"
              color="primary"
              sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}
            >
              <InfoIcon />
              {isArabic ? "البنود الإضافية" : "Additional Clauses"} (
              {sections.additional.length})
            </Typography>
            {sections.additional.length > 0 ? (
              <Paper elevation={2} sx={{ p: 2, bgcolor: "#e3f2fd" }}>
                <Stack spacing={1.5}>
                  {sections.additional.map((item, i) => (
                    <Box
                      key={i}
                      sx={{
                        p: 2,
                        bgcolor: "white",
                        borderRadius: 2,
                        borderLeft: "4px solid #2196f3",
                      }}
                    >
                      <Typography color="primary.dark">{item}</Typography>
                    </Box>
                  ))}
                </Stack>
              </Paper>
            ) : (
              <Alert severity="info">
                {isArabic ? "لا توجد بنود إضافية" : "No extra clauses found"}
              </Alert>
            )}
          </Box>

          <Divider />

          {/* Summary */}
          {sections.summary.length > 0 && (
            <Box>
              <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
                {isArabic ? "📋 الملخص" : "📋 Summary"}
              </Typography>
              <Paper
                elevation={3}
                sx={{
                  p: 3,
                  bgcolor: "#f5f5f5",
                  borderRadius: 2,
                }}
              >
                <Stack spacing={1}>
                  {sections.summary.map((line, i) => (
                    <Typography
                      key={i}
                      sx={{
                        fontSize: "0.95rem",
                        lineHeight: 1.8,
                        textAlign: isArabic ? "right" : "left",
                      }}
                    >
                      {line}
                    </Typography>
                  ))}
                </Stack>
              </Paper>
            </Box>
          )}
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
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 2,
                        width: "100%",
                      }}
                    >
                      <Typography
                        fontWeight="bold"
                        color={result.success ? "success.main" : "error.main"}
                        flex={1}
                      >
                        {result.name} → {result.success ? "Success" : "Failed"}
                      </Typography>
                      {result.success && (
                        <Button
                          variant="contained"
                          size="small"
                          startIcon={
                            translatingIndex === i ? (
                              <CircularProgress size={16} color="inherit" />
                            ) : (
                              <TranslateIcon />
                            )
                          }
                          onClick={(e) => {
                            e.stopPropagation();
                            handleTranslateReport(i);
                          }}
                          disabled={translatingIndex === i}
                          sx={{
                            background:
                              "linear-gradient(45deg, #FF6B6B 30%, #FFA500 90%)",
                            color: "white",
                            fontWeight: "bold",
                          }}
                        >
                          {result.isTranslated ? "مترجم ✓" : "ترجم للعربية"}
                        </Button>
                      )}
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    {result.isTranslated && result.translatedReport ? (
                      <Box>
                        <Alert severity="info" sx={{ mb: 2 }}>
                          النسخة العربية (مترجمة):
                        </Alert>
                        {renderBeautifulReport(result.translatedReport, true)}
                        <Divider sx={{ my: 3 }} />
                        <Alert severity="info" sx={{ mb: 2 }}>
                          Original English Version:
                        </Alert>
                        {renderBeautifulReport(result.report, false)}
                      </Box>
                    ) : (
                      renderBeautifulReport(result.report, false)
                    )}
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
