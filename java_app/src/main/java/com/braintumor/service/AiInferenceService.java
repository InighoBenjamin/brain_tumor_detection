package com.braintumor.service;

import com.braintumor.entity.AiPrediction;
import com.braintumor.entity.MriMetaData;
import com.braintumor.repository.AiPredictionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Calls the Python inference.py script via ProcessBuilder.
 * Runs asynchronously so the HTTP response returns immediately
 * while the AI prediction processes in the background.
 *
 * Python script output (stdout) must contain lines like:
 *   WT_DICE=0.8703
 *   TC_DICE=0.7448
 *   ET_DICE=0.6723
 *   TUMOR_DETECTED=1
 *   TUMOR_AREA_MM2=1245.5
 *   ESTIMATED_REGION=Frontal Lobe
 *   MASK_PATH=/path/to/mask.png
 *   HEATMAP_PATH=/path/to/heatmap.png
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiInferenceService {

    private final AiPredictionRepository predictionRepository;

    @Value("${app.python.executable}")
    private String pythonExecutable;

    @Value("${app.python.script}")
    private String inferenceScript;

    @Value("${app.model.checkpoint}")
    private String modelCheckpoint;

    @Value("${app.predictions.dir}")
    private String predictionsDir;

    /**
     * Triggers async AI inference for a given MRI record.
     * Creates a prediction record with status=processing, then runs Python.
     */
    @Async
    public void runInference(MriMetaData mri) {
        // Create prediction record with status = processing
        AiPrediction prediction = new AiPrediction();
        prediction.setMri(mri);
        prediction.setStatus(AiPrediction.Status.processing);
        prediction.setModelVersion("epoch35_wt0.8703");
        prediction = predictionRepository.save(prediction);

        try {
            Path outputDir = Paths.get(predictionsDir, "mri_" + mri.getMriId());

            List<String> command = new ArrayList<>();
            command.add(pythonExecutable);
            command.add(inferenceScript);
            command.add("--checkpoint");   command.add(modelCheckpoint);
            command.add("--h5_file");      command.add(mri.getMriPath());
            command.add("--output_dir");   command.add(outputDir.toString());
            command.add("--output_json");  // flag to print structured output

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // Parse stdout line by line
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );

            String line;
            while ((line = reader.readLine()) != null) {
                log.debug("[AI] {}", line);
                parseLine(prediction, line);
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                prediction.setStatus(AiPrediction.Status.done);
                log.info("AI inference done for MRI id={} WT={}", mri.getMriId(), prediction.getWtDice());
            } else {
                prediction.setStatus(AiPrediction.Status.failed);
                log.error("AI inference failed for MRI id={} exit={}", mri.getMriId(), exitCode);
            }

        } catch (Exception e) {
            prediction.setStatus(AiPrediction.Status.failed);
            log.error("AI inference exception for MRI id={}: {}", mri.getMriId(), e.getMessage());
        }

        predictionRepository.save(prediction);
    }

    /** Parse key=value lines printed by inference.py */
    private void parseLine(AiPrediction prediction, String line) {
        if (line.startsWith("WT_DICE=")) {
            prediction.setWtDice(Float.parseFloat(line.split("=")[1].trim()));
        } else if (line.startsWith("TC_DICE=")) {
            prediction.setTcDice(Float.parseFloat(line.split("=")[1].trim()));
        } else if (line.startsWith("ET_DICE=")) {
            prediction.setEtDice(Float.parseFloat(line.split("=")[1].trim()));
        } else if (line.startsWith("TUMOR_DETECTED=")) {
            prediction.setTumorDetected("1".equals(line.split("=")[1].trim()));
        } else if (line.startsWith("TUMOR_AREA_MM2=")) {
            prediction.setTumorAreaMm2(Float.parseFloat(line.split("=")[1].trim()));
        } else if (line.startsWith("ESTIMATED_REGION=")) {
            prediction.setEstimatedRegion(line.substring("ESTIMATED_REGION=".length()).trim());
        } else if (line.startsWith("MASK_PATH=")) {
            prediction.setMaskFilePath(line.substring("MASK_PATH=".length()).trim());
        } else if (line.startsWith("HEATMAP_PATH=")) {
            prediction.setHeatMapPath(line.substring("HEATMAP_PATH=".length()).trim());
        }
    }
}
