package com.braintumor.service;

import com.braintumor.entity.*;
import com.braintumor.repository.AiPredictionRepository;
import com.braintumor.repository.ReviewRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ReviewService {

    private final ReviewRepository reviewRepository;
    private final AiPredictionRepository predictionRepository;

    /** Get all pending predictions waiting for radiologist review */
    public List<AiPrediction> getPendingPredictions() {
        return predictionRepository.findByStatus(AiPrediction.Status.done);
    }

    /** Radiologist submits their review of an AI prediction */
    public RadiologistReview submitReview(
            Integer aiPredictionId,
            Integer radiologistId,
            String diagnosis,
            String notes,
            RadiologistReview.ReviewStatus status,
            String modifiedMaskPath
    ) {
        AiPrediction prediction = predictionRepository.findById(aiPredictionId)
            .orElseThrow(() -> new IllegalArgumentException("Prediction not found: " + aiPredictionId));

        Radiologist radiologist = new Radiologist();
        radiologist.setRadiologistId(radiologistId);

        RadiologistReview review = new RadiologistReview();
        review.setAiPrediction(prediction);
        review.setRadiologist(radiologist);
        review.setDiagnosis(diagnosis);
        review.setReviewNotes(notes);
        review.setStatus(status);
        review.setModifiedMaskFilePath(modifiedMaskPath);
        review.setReviewedAt(LocalDateTime.now());

        return reviewRepository.save(review);
    }

    public List<RadiologistReview> getReviewsByRadiologist(Integer radiologistId) {
        return reviewRepository.findByRadiologist_RadiologistId(radiologistId);
    }
}
