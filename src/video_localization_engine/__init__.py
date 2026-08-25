"""Video Localization Engine (VLE).

通用视频本地化引擎。L1-L7 分层架构, 见 video_localization_engine.ARCHITECTURE.

数据流向:
  L1 VideoAnalyzer  ->  FramePacket stream
  L2 RegionPolicies ->  RegionProposal (per frame)
  L3 Detector       ->  TextCandidate (per frame)
  L4 Manager        ->  SubtitleInstance (跨帧稳定身份)
  L5 MaskGenerator  ->  binary mask (per frame)
  L6 InpaintingEng  ->  inpainted frame (per frame)
  L7 Localizer      ->  target-lang subtitle + audio + composited video (后续)
"""
