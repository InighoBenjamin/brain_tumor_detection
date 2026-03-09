package com.braintumor.repository;

import com.braintumor.entity.RadiologistReview;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface ReviewRepository extends JpaRepository<RadiologistReview, Integer> {
    List<RadiologistReview> findByRadiologist_RadiologistId(Integer radiologistId);
    List<RadiologistReview> findByStatus(RadiologistReview.ReviewStatus status);
}
