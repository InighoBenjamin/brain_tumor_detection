package com.braintumor.controller;

import com.braintumor.entity.AiPrediction;
import com.braintumor.entity.RadiologistReview;
import com.braintumor.repository.AiPredictionRepository;
import com.braintumor.service.ReviewService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/review")
@RequiredArgsConstructor
public class ReviewController {

    private final ReviewService reviewService;
    private final AiPredictionRepository predictionRepository;

    /**
     * GET /api/review/pending
     * List all AI predictions waiting for radiologist review.
     */
    @GetMapping("/pending")
    @PreAuthorize("hasAnyRole('radiologist','admin')")
    public ResponseEntity<List<AiPrediction>> getPending() {
        return ResponseEntity.ok(reviewService.getPendingPredictions());
    }

    /**
     * GET /api/review/prediction/{id}
     * Get a single AI prediction by ID.
     */
    @GetMapping("/prediction/{id}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<AiPrediction> getPrediction(@PathVariable Integer id) {
        return ResponseEntity.ok(
            predictionRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Prediction not found: " + id))
        );
    }

    /**
     * POST /api/review/submit
     * Radiologist submits their review of an AI prediction.
     *
     * Body (JSON):
     * {
     *   "aiPredictionId": 1,
     *   "radiologistId": 2,
     *   "diagnosis": "High-grade glioma in right frontal lobe",
     *   "notes": "AI mask slightly underestimates edema region",
     *   "status": "approved"
     * }
     */
    @PostMapping("/submit")
    @PreAuthorize("hasAnyRole('radiologist','admin')")
    public ResponseEntity<RadiologistReview> submit(@RequestBody Map<String, Object> body) {
        RadiologistReview review = reviewService.submitReview(
            (Integer) body.get("aiPredictionId"),
            (Integer) body.get("radiologistId"),
            (String)  body.get("diagnosis"),
            (String)  body.get("notes"),
            RadiologistReview.ReviewStatus.valueOf((String) body.get("status")),
            (String)  body.get("modifiedMaskPath")
        );
        return ResponseEntity.ok(review);
    }

    /**
     * GET /api/review/radiologist/{id}
     * Get all reviews by a specific radiologist.
     */
    @GetMapping("/radiologist/{id}")
    @PreAuthorize("hasAnyRole('radiologist','admin')")
    public ResponseEntity<List<RadiologistReview>> getByRadiologist(@PathVariable Integer id) {
        return ResponseEntity.ok(reviewService.getReviewsByRadiologist(id));
    }
}
