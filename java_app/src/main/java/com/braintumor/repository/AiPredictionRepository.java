package com.braintumor.repository;

import com.braintumor.entity.AiPrediction;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;

public interface AiPredictionRepository extends JpaRepository<AiPrediction, Integer> {
    Optional<AiPrediction> findByMri_MriId(Integer mriId);
    List<AiPrediction> findByStatus(AiPrediction.Status status);
}
