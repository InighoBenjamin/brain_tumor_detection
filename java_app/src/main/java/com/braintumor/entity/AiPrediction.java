package com.braintumor.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "ai_predictions")
public class AiPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer aiPredictionsId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "mri_id", nullable = false)
    private MriMetaData mri;

    @Column(length = 255)
    private String maskFilePath;       // segmentation mask image path

    @Column(length = 255)
    private String heatMapPath;        // overlay visualization path

    private Float wtDice;              // Whole Tumor Dice
    private Float tcDice;              // Tumor Core Dice
    private Float etDice;              // Enhancing Tumor Dice

    @Column(nullable = false)
    private Boolean tumorDetected = false;

    private Float tumorAreaMm2;

    @Column(length = 50)
    private String estimatedRegion;    // e.g. "Frontal Lobe"

    @Column(length = 50)
    private String modelVersion;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Status status = Status.processing;

    @Column(updatable = false)
    private LocalDateTime predictedAt = LocalDateTime.now();

    public enum Status { processing, done, failed }
}
