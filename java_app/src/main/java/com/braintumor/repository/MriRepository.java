package com.braintumor.repository;

import com.braintumor.entity.MriMetaData;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface MriRepository extends JpaRepository<MriMetaData, Integer> {
    List<MriMetaData> findByPatient_PatientId(Integer patientId);
}
