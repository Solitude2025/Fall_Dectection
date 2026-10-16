package com.falldetection.event;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.Instant;
import java.util.List;

public interface FallEventRepository extends JpaRepository<FallEvent, String> {
    List<FallEvent> findAllByOrderByCreatedAtDesc(Pageable pageable);
    List<FallEvent> findByStatusOrderByCreatedAtDesc(String status, Pageable pageable);
    long countByStatus(String status);
    long countByCreatedAtAfter(Instant after);
}
