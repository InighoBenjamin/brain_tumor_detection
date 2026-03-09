package com.braintumor.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "radiologist_review")
public class RadiologistReview {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer reviewId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "ai_predictions_id", nullable = false)
    private AiPrediction aiPrediction;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "radiologist_id", nullable = false)
    private Radiologist radiologist;

    @Column(length = 255)
    private String modifiedMaskFilePath;  // if radiologist corrects the AI mask

    @Column(length = 200)
    private String diagnosis;

    @Column(columnDefinition = "TEXT")
    private String reviewNotes;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ReviewStatus status = ReviewStatus.pending;

    private LocalDateTime reviewedAt;

    @Column(updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    public enum ReviewStatus { pending, approved, rejected }
}
