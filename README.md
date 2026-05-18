# gomunamu.github.io

개인 기술 블로그 저장소입니다. Jekyll과 Chirpy 테마를 기반으로 운영하며, 시계열 분석과 예측 모델링을 중심으로 공부한 내용을 정리합니다.

## 블로그 주제

주로 다음 내용을 다룹니다.

- 시계열 분석의 기본 개념: 정상성, 단위근, 차분, 자기상관
- 고전적 예측 모델: ARIMA, SARIMA, ETS
- 실무형 예측 도구: Prophet
- Python 기반 실습 코드와 결과 해석
- 모델 선택, 잔차 진단, 예측구간, backtesting 등 예측 모델링에서 자주 놓치는 부분

글은 가능하면 다음 흐름을 따릅니다.

1. 직관
2. 수식
3. 코드
4. 진단
5. 한계와 주의점

## 로컬 실행

Ruby와 Bundler가 설치되어 있다면 다음 명령으로 로컬에서 확인할 수 있습니다.

```bash
bundle install
bundle exec jekyll serve
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:4000
```

정적 사이트 빌드는 다음 명령으로 실행합니다.

```bash
bundle exec jekyll build
```

## 주요 디렉터리

```text
_posts/        블로그 게시글
_tabs/         About 등 상단/사이드 탭 페이지
assets/        이미지, CSS, 기타 정적 자원
_config.yml    Jekyll/Chirpy 설정
```

## 작성 원칙

- 개념 설명은 지나치게 단정하지 않고, 필요한 가정과 한계를 함께 적습니다.
- 예측 결과는 점예측뿐 아니라 예측구간과 잔차 진단을 함께 확인합니다.
- AIC/BIC 같은 정보기준은 보조 지표로 보고, 가능하면 검증 구간 성능도 함께 봅니다.
- 코드 예시는 재현 가능하고 독자가 따라 실행할 수 있도록 작성합니다.

## 기반 테마

이 블로그는 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) Jekyll 테마를 사용합니다.

## License

블로그 콘텐츠의 저작권은 저장소 소유자에게 있습니다. 테마와 템플릿 관련 코드는 각 원저작자의 라이선스를 따릅니다.
